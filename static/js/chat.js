// static/js/chat.js
import * as api from './api.js';
import * as ui from './ui.js';
import { getToken } from './auth.js';

let currentConversationId = null;
let chatMessages = []; // Stores {role, content, images (as dataURLs for UI, consistent for API)}
let elements = {};
let imagePreviewHandler = { getFiles: () => [], clear: () => {}, addFiles: () => {} };
let lastSentUserMessageForRetry = null; 

let currentAbortController = null;
let isSendingMessage = false;


// Helper to convert File object to Data URL (for UI rendering & API)
async function fileToDataURL(file) {
    return new Promise((resolve, reject) => {
        if (!(file instanceof File)) {
            if (typeof file === 'string' && file.startsWith('data:')) { // Allow any data URL string
                if (file.includes(',')) {
                    resolve(file);
                } else if (/^data:[^;]+;base64,$/.test(file) && file.endsWith(',')) { // Handles data:mime/type;base64, (empty content)
                    resolve(file);
                } else {
                    reject(new Error(`Malformed data URL string (e.g., missing comma or invalid format) for fileToDataURL: ${file.substring(0,50)}`));
                }
                return;
            }
            reject(new Error(`Invalid input for fileToDataURL: not a File or Data URL string. Input type: ${typeof file}, value: ${String(file).substring(0,30)}`));
            return;
        }
        // It's a File object
        const reader = new FileReader();
        reader.onload = () => {
            if (typeof reader.result === 'string' && reader.result.startsWith('data:')) {
                 if (reader.result.includes(',')) {
                    resolve(reader.result);
                } else if (/^data:[^;]+;base64,$/.test(reader.result) && reader.result.endsWith(',')) { // Valid empty data:mime/type;base64,
                    resolve(reader.result);
                } else {
                     reject(new Error(`FileReader produced a malformed data URL: ${reader.result.substring(0,50)}`));
                }
            } else {
                reject(new Error(`FileReader did not produce a string data URL. Result type: ${typeof reader.result}`));
            }
        };
        reader.onerror = error => reject(error);
        reader.readAsDataURL(file);
    });
}

// Helper to convert File object or Data URL to PURE Base64 string (e.g. for Ollama /api/generate if needed elsewhere)
async function fileToPureBase64(fileOrDataUrl) {
    return new Promise(async (resolve, reject) => {
        try {
            let dataUrlToProcess;
            if (fileOrDataUrl instanceof File) {
                dataUrlToProcess = await fileToDataURL(fileOrDataUrl); 
            } else if (typeof fileOrDataUrl === 'string' && fileOrDataUrl.startsWith('data:')) {
                dataUrlToProcess = fileOrDataUrl;
            } else if (typeof fileOrDataUrl === 'string') {
                resolve(fileOrDataUrl);
                return;
            } else {
                reject(new Error("Invalid input for fileToPureBase64: not a File, Data URL, or base64 string."));
                return;
            }

            if (!dataUrlToProcess.startsWith('data:image')) {
                reject(new Error(`Cannot convert non-image data URL to pure base64: ${dataUrlToProcess.substring(0,60)}...`));
                return;
            }

            const parts = dataUrlToProcess.split(',');
            if (parts.length === 2) { 
                resolve(parts[1]);
            } else {
                reject(new Error(`Malformed data URL for pure base64 extraction (expected 1 comma, found ${parts.length - 1}): ${dataUrlToProcess.substring(0,60)}...`));
            }
        } catch (error) {
            reject(new Error(`fileToPureBase64 processing error: ${error.message}`));
        }
    });
}


export function initChatElements(domElements) {
    elements = domElements;
    imagePreviewHandler = ui.handleImagePreviews(elements.imageInputElement, elements.imagePreviewsContainer);
    elements.chatFormElement.addEventListener('submit', (e) => handleSendMessage(e, false));
    elements.messageInputElement.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSendMessage(e, false); }
    });
    elements.messageInputElement.addEventListener('paste', async (event) => {
        const items = (event.clipboardData || event.originalEvent.clipboardData)?.items;
        if (items) {
            const filesToProcess = [];
            for (let i = 0; i < items.length; i++) {
                if (items[i].type.indexOf('image') !== -1) {
                    const file = items[i].getAsFile();
                    if (file) filesToProcess.push(file);
                }
            }
            if (filesToProcess.length > 0) {
                event.preventDefault();
                imagePreviewHandler.addFiles(filesToProcess);
            }
        }
    });
    initForkConversation();
}

export function resetChatState() {
    currentConversationId = null; chatMessages = []; lastSentUserMessageForRetry = null;
    if (elements.chatMessagesElement) elements.chatMessagesElement.innerHTML = '';
    if (elements.conversationsListElement) {
        const activeItem = elements.conversationsListElement.querySelector('[style*="background-color: var(--bg-tertiary)"]');
        if (activeItem) { activeItem.style.backgroundColor = ''; activeItem.style.fontWeight = ''; }
    }
    if (elements.conversationTitleElement) elements.conversationTitleElement.textContent = 'New Chat';
    if (elements.messageInputElement) elements.messageInputElement.value = '';
    imagePreviewHandler.clear();
    ui.updateActionButtonsVisibility(null, []);
    window.location.hash = '';
    if (currentAbortController) {
        currentAbortController.abort();
        currentAbortController = null;
    }
    isSendingMessage = false;
    ui.showLoadingIndicator(false, elements.sendButton, "Send");
}

export async function loadInitialChatData() {
    if (!getToken()) return;
    ui.showLoadingIndicator(true, null, "Initializing...");
    try {
        await Promise.all([loadConversationsList(), loadModelsAndProfiles()]);
        const hash = window.location.hash.substring(1);
        if (hash) await loadConversation(hash, false);
        else startNewChat(false);
    } catch (error) { console.error("Error loading initial chat data:", error); }
    finally { ui.showLoadingIndicator(false); }
}

async function loadModelsAndProfiles() {
    try {
        const [ollamaModelsRes, openAIModelsRes, profilesRes] = await Promise.allSettled([
            api.fetchOllamaModels(), api.fetchOpenAIModels(), api.fetchRequestProfiles()
        ]);
        const ollamaModels = ollamaModelsRes.status === 'fulfilled' ? (ollamaModelsRes.value.models || []) : [];
        const openAIModels = openAIModelsRes.status === 'fulfilled' ? (openAIModelsRes.value.models || []) : [];
        const profiles = profilesRes.status === 'fulfilled' ? (profilesRes.value || []) : [];
        const allModels = [...ollamaModels, ...openAIModels].filter(m => m && m.NAME).sort((a, b) => a.NAME.localeCompare(b.NAME));
        
        ui.populateSelect(elements.modelSelectElement, allModels);
        ui.populateSelect(elements.backupModelSelectElement, allModels, true); // Add "None" option for backup
        ui.populateSelect(elements.profileSelectElement, profiles, true); // Add "None" option for profiles

        const defaultModel = "o3-mini-high";
        if (elements.modelSelectElement.querySelector(`option[value="${defaultModel}"]`)) {
            elements.modelSelectElement.value = defaultModel;
        } else if (allModels.length > 0) {
            elements.modelSelectElement.value = allModels[0].NAME;
        }

        // Default backup model to "gpt-4.1" if available, otherwise "None"
        const defaultBackupModel = "gpt-4.1"; // Define your desired default backup model
        if (elements.backupModelSelectElement) {
            const backupModelOption = elements.backupModelSelectElement.querySelector(`option[value="${defaultBackupModel}"]`);
            if (backupModelOption) {
                elements.backupModelSelectElement.value = defaultBackupModel;
            } else {
                // Fallback to "None" if "gpt-4.1" is not in the list
                elements.backupModelSelectElement.value = ""; 
            }
        }

    } catch (error) { console.error("Error loading models/profiles:", error); }
}

export async function loadConversationsList() {
    if (!getToken()) return;
    try {
        const data = await api.fetchConversations();
        ui.updateConversationList(data.conversations, elements.conversationsListElement, handleConversationSelection, currentConversationId);
    } catch (error) {
        console.error("Error fetching conversations:", error);
        if(elements.conversationsListElement) elements.conversationsListElement.innerHTML = '<li class="conversation-preview" style="color: var(--accent-danger);">Failed to load.</li>';
    }
}

export async function loadConversation(conversationId, showLoader = true) {
    if (!conversationId) return startNewChat(showLoader);
    if (showLoader) ui.showLoadingIndicator(true, null, "Loading Chat...");
    try {
        const data = await api.fetchConversationMessages(conversationId);
        currentConversationId = conversationId; lastSentUserMessageForRetry = null;
        chatMessages = data.messages.map(msg => ({ role: msg.role, content: msg.content || "", images: msg.images || null }));
        ui.displayChatMessages(chatMessages, elements.chatMessagesElement);
        const convosData = await api.fetchConversations();
        const currentConvoData = convosData.conversations.find(c => c.conversation_id === conversationId);
        if (elements.conversationTitleElement) elements.conversationTitleElement.textContent = currentConvoData?.preview || `Chat ${conversationId.substring(0, 8)}`;
        window.location.hash = conversationId;
        ui.updateConversationList(convosData.conversations, elements.conversationsListElement, handleConversationSelection, currentConversationId);
    } catch (error) {
        console.error(`Error loading conversation ${conversationId}:`, error);
        alert(`Failed to load: ${error.message}.`); startNewChat(false);
    } finally {
        if (showLoader) ui.showLoadingIndicator(false);
        ui.updateActionButtonsVisibility(currentConversationId, chatMessages);
    }
}

export async function handleConversationSelection(conversationId) {
    if (elements.sidebar && window.innerWidth <= 768 && elements.sidebar.classList.contains('open')) {
        elements.sidebar.classList.remove('open'); elements.sidebar.style.left = '-280px';
    }
    if (conversationId !== currentConversationId) await loadConversation(conversationId);
}

export function startNewChat(showLoader = true) {
    if (showLoader && elements.sendButton) ui.showLoadingIndicator(true, elements.sendButton, "New Chat...", false); // Not abortable
    resetChatState();
    if (elements.messageInputElement) elements.messageInputElement.focus();
    if (getToken()) loadConversationsList();
    else if(elements.conversationsListElement) elements.conversationsListElement.innerHTML = '<li class="conversation-preview" style="color: var(--text-secondary); text-align: center;">Please sign in.</li>';
    if (showLoader && elements.sendButton) ui.showLoadingIndicator(false, elements.sendButton);
}


async function handleSendMessage(event, isRetryAttempt = false, messageToRetry = null) {
    if (event) event.preventDefault();
    
    if (isSendingMessage) {
        if (elements.sendButton.dataset.canAbort === 'true' && currentAbortController) {
            console.log("Attempting to abort request...");
            currentAbortController.abort();
            // The catch block for fetch will handle UI updates and state reset.
        } else {
            console.log("Message send in progress or abort not ready, click ignored.");
        }
        return; 
    }

    let userMessageContent;
    let imagesForPayloadAndUI = []; 

    if (isRetryAttempt && messageToRetry) {
        userMessageContent = messageToRetry.content;
        imagesForPayloadAndUI = messageToRetry.images || [];
        ui.addChatMessage({role: 'user', content: userMessageContent, images: imagesForPayloadAndUI}, elements.chatMessagesElement, true);
    } else {
        userMessageContent = elements.messageInputElement.value.trim();
        const attachedFiles = imagePreviewHandler.getFiles(); 
        if (!userMessageContent && attachedFiles.length === 0) return;

        for (const file of attachedFiles) {
            try {
                const dataUrl = await fileToDataURL(file);
                imagesForPayloadAndUI.push(dataUrl);
            } catch (error) { 
                console.error("Error processing file to DataURL:", error); 
                alert(`Error processing image: ${error.message}. Please try a different image or check console.`); 
                return; 
            }
        }
        ui.addChatMessage({role: 'user', content: userMessageContent, images: imagesForPayloadAndUI}, elements.chatMessagesElement, true);
        lastSentUserMessageForRetry = { role: 'user', content: userMessageContent, images: imagesForPayloadAndUI }; 
        elements.messageInputElement.value = '';
        imagePreviewHandler.clear();
    }

    const userMessageForApiState = { role: 'user', content: userMessageContent, images: imagesForPayloadAndUI.length > 0 ? imagesForPayloadAndUI : null };

    isSendingMessage = true;
    currentAbortController = new AbortController();
    
    ui.showLoadingIndicator(true, elements.sendButton, isRetryAttempt ? "Retrying..." : "Sending...", true); // isAbortable = true
    
    const resolvedCurrentMessages = chatMessages.map(msg => msg); 
    const messagesForApi = [...resolvedCurrentMessages, userMessageForApiState];

    const payload = {
        messages: messagesForApi,
        stream: elements.streamToggleElement.checked,
        conversation_id: currentConversationId,
        image_b64: imagesForPayloadAndUI.length > 0 ? imagesForPayloadAndUI : null, // Send null if no images
    };

    const selectedModelName = elements.modelSelectElement.value;
    if (elements.profileSelectElement.value) {
        payload.request_profile = elements.profileSelectElement.value;
    } else {
        payload.model = selectedModelName;
        const backupModelValue = elements.backupModelSelectElement.value; // Get single value
        if (backupModelValue) { // Check if a backup model is selected (not "None")
            payload.backup_models = backupModelValue; // Singular key
        }
    }
    let responseSuccessful = false;
    try {
        const response = await api.fetchWithAuth('/chat-completion', { 
            method: 'POST', 
            body: JSON.stringify(payload),
            signal: currentAbortController.signal
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: "Request failed: " + response.status }));
            throw new Error(errorData.detail || `Error: ${response.status}`);
        }
        
        responseSuccessful = true; // Mark as successful before processing stream/json
        const userMessageToStore = {role: 'user', content: userMessageContent, images: imagesForPayloadAndUI.length > 0 ? imagesForPayloadAndUI : null};
        chatMessages.push(userMessageToStore);
        lastSentUserMessageForRetry = null;

        const newConvIdFromServer = response.headers.get('X-Conversation-ID');
        let conversationJustCreatedOrUpdated = false;
        if (newConvIdFromServer && (!currentConversationId || currentConversationId !== newConvIdFromServer)) {
            currentConversationId = newConvIdFromServer; window.location.hash = currentConversationId; conversationJustCreatedOrUpdated = true;
        } else if (currentConversationId) { conversationJustCreatedOrUpdated = true; }

        if (payload.stream) {
            let fullAssistantResponse = ""; const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = "";
            while (true) {
                const { value, done } = await reader.read(); if (done) break;
                buffer += decoder.decode(value, { stream: true }); let boundary;
                while ((boundary = buffer.indexOf('\n')) >= 0) {
                    const line = buffer.substring(0, boundary).trim(); buffer = buffer.substring(boundary + 1);
                    if (line) { try { const chunk = JSON.parse(line); if (chunk.message !== undefined) { fullAssistantResponse += chunk.message; ui.updateStreamingMessage(fullAssistantResponse, elements.chatMessagesElement); } else if (chunk.error) { throw new Error(JSON.stringify(chunk.error));}} catch (e) { console.warn("Stream parse error:", line, e);}}
                }
            }
            if (buffer.trim()) { try { const chunk = JSON.parse(buffer.trim()); if (chunk.message !== undefined) {fullAssistantResponse += chunk.message; ui.updateStreamingMessage(fullAssistantResponse, elements.chatMessagesElement);} } catch(e) { console.warn("Final stream chunk error:", buffer.trim(), e);}}
            ui.finalizeStreamingMessage(elements.chatMessagesElement);
            if (fullAssistantResponse.trim()) {
                 chatMessages.push({ role: 'assistant', content: fullAssistantResponse, images: null });
            }
        } else {
            const data = await response.json();
            if (data.message !== undefined && data.message.trim()) {
                const assistantMessage = { role: 'assistant', content: data.message, images: null };
                ui.addChatMessage(assistantMessage, elements.chatMessagesElement, false);
                chatMessages.push(assistantMessage);
            }
        }
        ui.updateActionButtonsVisibility(currentConversationId, chatMessages);
        if (conversationJustCreatedOrUpdated) await loadConversationsList();

    } catch (error) {
        if (error.name === 'AbortError') {
            console.log('Request aborted by user.');
            ui.addChatMessage({role: 'assistant', content: '[Request Cancelled by User]'}, elements.chatMessagesElement, false);
            // lastSentUserMessageForRetry is still set, user can retry if they wish.
            // If the user message was optimistically added, consider removing it or marking as cancelled.
            // For now, we keep the user message and add a "Cancelled" response.
        } else {
            console.error("Error sending/receiving message:", error);
            const messageThatFailed = isRetryAttempt ? messageToRetry : lastSentUserMessageForRetry;
            
            const userMessageUIElements = Array.from(elements.chatMessagesElement.querySelectorAll('.message-group.user'));
            const lastUserMessageUI = userMessageUIElements[userMessageUIElements.length -1];

            if(messageThatFailed && lastUserMessageUI && messageThatFailed.content !== undefined) {
                const uiContentElement = lastUserMessageUI.querySelector('.chat-message.user div:not(.user-images)');
                const uiContent = uiContentElement ? uiContentElement.textContent || "" : "";
                const uiImageCount = lastUserMessageUI.querySelectorAll('.user-images img').length;
                const failedImageCount = messageThatFailed.images ? messageThatFailed.images.length : 0;
                const contentToCompare = messageThatFailed.content || "";
                if (uiContent.startsWith(contentToCompare.substring(0,30)) && uiImageCount === failedImageCount) {
                    lastUserMessageUI.remove(); // Remove the optimistically added user message that failed
                }
            }
            
            if (!isRetryAttempt && messageThatFailed && !responseSuccessful) { // only keep for retry if not successful
                lastSentUserMessageForRetry = messageThatFailed;
            } else if (responseSuccessful && !isRetryAttempt) { // If it was successful but then failed in stream/json processing
                 lastSentUserMessageForRetry = null; // Don't offer retry for this part
            }


            const retryCb = lastSentUserMessageForRetry ? () => handleSendMessage(null, true, lastSentUserMessageForRetry) : null;
            ui.addErrorMessageWithRetry({ role: 'assistant', content: `Error: ${error.message}.` }, elements.chatMessagesElement, retryCb);
        }
    
    } finally {
        isSendingMessage = false;
        currentAbortController = null;
        ui.showLoadingIndicator(false, elements.sendButton, "Send");
    }
}


function initForkConversation() {
    if (elements.forkConversationBtnElement) {
        elements.forkConversationBtnElement.addEventListener('click', () => {
            if (!currentConversationId || chatMessages.filter(msg => msg.role !== 'system').length === 0) { alert("No messages to fork."); return; }
            ui.populateForkModal(chatMessages, elements.forkMessageListElement, handleForkSelection);
            ui.showElement(elements.forkModalElement);
        });
    }
    if (elements.cancelForkBtnElement) {
        elements.cancelForkBtnElement.addEventListener('click', () => ui.hideElement(elements.forkModalElement));
    }
}

async function handleForkSelection(event) {
    const selectedMessageOriginalIndex = parseInt(event.currentTarget.dataset.messageIndex);
    if (isNaN(selectedMessageOriginalIndex) || !currentConversationId) { alert("Invalid selection for fork."); return; }
    ui.hideElement(elements.forkModalElement);
    ui.showLoadingIndicator(true, null, "Forking...");
    try {
        const result = await api.forkConversationApi(currentConversationId, selectedMessageOriginalIndex);
        currentConversationId = result.new_conversation_id; window.location.hash = currentConversationId;
        await loadConversation(currentConversationId, false); await loadConversationsList();
    } catch (error) { console.error('Error forking:', error); alert(`Could not fork: ${error.message}`);
    } finally { ui.showLoadingIndicator(false); }
}

export function exportChat() {
    const displayableMessages = chatMessages.filter(msg => msg.role !== 'system');
    if (displayableMessages.length === 0) {
        alert("No messages to export.");
        return;
    }

    const modelInUse = elements.profileSelectElement.value 
        ? `profile:${elements.profileSelectElement.value}`
        : elements.modelSelectElement.value;
    
    const backupModelInUse = elements.backupModelSelectElement.value; // Single value

    const exportObj = {
        exported_at: new Date().toISOString(),
        current_conversation_id: currentConversationId,
        ui_selections: {
            model: modelInUse,
            backup_model: backupModelInUse || null, // Ensure null if empty
            stream: elements.streamToggleElement.checked,
        },
        messages: chatMessages.map(msg => ({
             role: msg.role,
             content: msg.content || "",
             images: msg.images || null 
        }))
    };

    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(exportObj, null, 2));
    const downloadAnchorNode = document.createElement('a');
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = currentConversationId 
        ? `conversation_${currentConversationId.substring(0,8)}_${timestamp}.json`
        : `new_chat_${timestamp}.json`;
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", filename);
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
}