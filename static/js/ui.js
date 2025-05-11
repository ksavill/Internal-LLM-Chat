// static/js/ui.js
import { THEME_KEY } from './config.js';

function applyTheme(isDark) {
    const hljsThemeLink = document.getElementById('hljs-theme');
    if (isDark) {
        document.documentElement.classList.add('dark');
        localStorage.setItem(THEME_KEY, 'dark');
        if (hljsThemeLink) hljsThemeLink.href = "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css";
    } else {
        document.documentElement.classList.remove('dark');
        localStorage.setItem(THEME_KEY, 'light');
        if (hljsThemeLink) hljsThemeLink.href = "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css";
    }
}

export function setupThemeToggle(toggleButtons) {
    const storedTheme = localStorage.getItem(THEME_KEY);
    if (storedTheme === 'dark') applyTheme(true);
    else if (storedTheme === 'light') applyTheme(false);
    else if (document.documentElement.classList.contains('dark') || (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches)) applyTheme(true);
    else applyTheme(false);

    toggleButtons.forEach(button => {
        if (button) button.addEventListener('click', () => applyTheme(!document.documentElement.classList.contains('dark')));
    });
}

export function showElement(element, displayType = null) {
    if (!element) return;
    if (element.classList.contains('modal')) element.classList.add('active');
    else {
        element.classList.remove('hidden');
        if (displayType && (element.style.display === 'none' || element.style.display === '')) element.style.display = displayType;
        else if (displayType) element.style.display = displayType;
    }
}
export function hideElement(element) {
    if (!element) return;
    if (element.classList.contains('modal')) element.classList.remove('active');
    else element.classList.add('hidden');
}

export function clearMessages(elements) { elements.forEach(el => { if (el) el.textContent = ''; }); }
export function displayAuthError(element, message) { if (element) element.textContent = message; }
export function displayAuthMessage(element, message) { if (element) element.textContent = message; }

export function populateSelect(selectElement, options, includeNone = false, valueKey = 'NAME', textKey = 'NAME') {
    if (!selectElement) return;
    selectElement.innerHTML = '';
    if (includeNone) {
        const noneOption = document.createElement('option'); noneOption.value = ''; noneOption.textContent = 'None';
        selectElement.appendChild(noneOption);
    }
    options.forEach(opt => {
        const option = document.createElement('option');
        const val = typeof opt === 'string' ? opt : opt[valueKey];
        const txt = typeof opt === 'string' ? opt : opt[textKey];
        if (val === undefined || txt === undefined) return;
        option.value = val; option.textContent = txt;
        selectElement.appendChild(option);
    });
}

export function updateConversationList(conversations, listElement, selectionHandler, currentConversationId) {
    if (!listElement) return;
    listElement.innerHTML = '';
    if (conversations.length === 0) {
        listElement.innerHTML = '<li class="conversation-preview" style="color: var(--text-secondary); text-align: center;">No conversations.</li>';
        return;
    }
    conversations.forEach(convo => {
        const item = document.createElement('li'); item.className = 'conversation-preview';
        if (convo.conversation_id === currentConversationId) {
            item.style.backgroundColor = 'var(--bg-tertiary)'; item.style.fontWeight = '600';
        }
        const p = document.createElement('p'); p.textContent = convo.preview || `Chat ${convo.conversation_id.substring(0,8)}`;
        const small = document.createElement('small'); small.textContent = new Date(convo.updated_at).toLocaleString([], {dateStyle: 'short', timeStyle: 'short'});
        item.appendChild(p); item.appendChild(small);
        item.dataset.id = convo.conversation_id;
        item.addEventListener('click', () => selectionHandler(convo.conversation_id));
        listElement.appendChild(item);
    });
}

function copyCodeToClipboard(preElement, event) {
    const codeElement = preElement.querySelector('code');
    if (!codeElement) return;
    const codeToCopy = codeElement.innerText.trim();
    navigator.clipboard.writeText(codeToCopy).then(() => {
        if (event && event.target) {
            const originalButton = event.target;
            const originalText = originalButton.dataset.originalCopyText || originalButton.textContent;
            if (!originalButton.dataset.originalCopyText) originalButton.dataset.originalCopyText = originalText;
            originalButton.textContent = 'Copied!';
            setTimeout(() => { originalButton.textContent = originalText; }, 1500);
        }
    }).catch(err => { console.error('Failed to copy code: ', err); alert('Failed to copy code.'); });
}

function processAndAddCodeBlockFunctionality(contentElement) {
    contentElement.querySelectorAll('pre').forEach(preElement => {
        if (preElement.parentElement.classList.contains('message-code-block-wrapper')) {
            const codeEl = preElement.querySelector('code');
            if (codeEl && typeof hljs !== 'undefined' && !codeEl.classList.contains('hljs') && codeEl.textContent.length > 0) {
                 try { hljs.highlightElement(codeEl); } catch(e) { console.warn("Highlight.js error:", e); }
            }
            return;
        }
        const codeElement = preElement.querySelector('code');
        if (!codeElement || !codeElement.textContent.trim()) return;

        const langMatch = codeElement.className.match(/language-(\S+)/);
        const lang = langMatch ? langMatch[1] : 'text';
        const wrapper = document.createElement('div'); wrapper.className = 'message-code-block-wrapper';
        const topCopyBtn = document.createElement('button'); topCopyBtn.textContent = 'Copy'; topCopyBtn.className = 'copy-btn-top';
        topCopyBtn.onclick = (event) => copyCodeToClipboard(preElement, event);
        const bottomBar = document.createElement('div'); bottomBar.className = 'code-block-bottom-bar';
        const bottomCopyBtn = document.createElement('button'); bottomCopyBtn.className = 'copy-btn-bottom'; bottomCopyBtn.textContent = `Copy ${lang}`;
        bottomCopyBtn.onclick = (event) => copyCodeToClipboard(preElement, event);
        bottomBar.appendChild(bottomCopyBtn);
        preElement.parentNode.insertBefore(wrapper, preElement);
        wrapper.appendChild(topCopyBtn); wrapper.appendChild(preElement); wrapper.appendChild(bottomBar);
        if (typeof hljs !== 'undefined') {
            try { hljs.highlightElement(codeElement); } catch(e) { console.warn("Highlight.js error:", e); }
        }
    });
}

export function renderMessage(message, chatMessagesElement, isUserMessage) {
    if (message.role === 'system') return; // System messages are not typically rendered in the chat window itself
    if (!chatMessagesElement) return;
    const messageGroup = document.createElement('div'); messageGroup.className = `message-group ${isUserMessage ? 'user' : 'bot'}`;
    const avatar = document.createElement('div'); avatar.className = isUserMessage ? 'user-avatar' : 'bot-avatar'; avatar.textContent = isUserMessage ? 'U' : 'B';
    const messagesContainer = document.createElement('div'); messagesContainer.className = 'messages';
    const messageBubble = document.createElement('div'); messageBubble.className = `chat-message ${isUserMessage ? 'user' : 'bot'}`;
    if (message.images && message.images.length > 0) {
        const imagesDiv = document.createElement('div'); imagesDiv.className = 'user-images';
        message.images.forEach(imageData => { // imageData is expected to be a DataURL
            const img = document.createElement('img');
            if (typeof imageData === 'string' && imageData.startsWith('data:image')) {
                img.src = imageData;
            } else if (typeof imageData === 'string') { 
                // Fallback if it's somehow not a data:image URL but a pure base64 string, though this shouldn't happen with current logic
                console.warn("Rendering image from non-DataURL string, attempting to prefix for display:", String(imageData).substring(0,30));
                img.src = `data:image/jpeg;base64,${imageData}`; 
            } else {
                console.warn("Invalid image data for rendering:", imageData);
            }
            imagesDiv.appendChild(img);
        });
        messageBubble.appendChild(imagesDiv);
    }
    const contentWrapper = document.createElement('div');
    if (message.content) {
        if (typeof marked !== 'undefined') {
            contentWrapper.innerHTML = marked.parse(message.content || '');
            processAndAddCodeBlockFunctionality(contentWrapper);
        } else {
            const p = document.createElement('p'); p.textContent = message.content || ''; contentWrapper.appendChild(p);
        }
    }
    messageBubble.appendChild(contentWrapper);
    messagesContainer.appendChild(messageBubble);
    if (isUserMessage) { messageGroup.appendChild(messagesContainer); messageGroup.appendChild(avatar); }
    else { messageGroup.appendChild(avatar); messageGroup.appendChild(messagesContainer); }
    chatMessagesElement.appendChild(messageGroup);
    chatMessagesElement.scrollTop = chatMessagesElement.scrollHeight;
}

export function displayChatMessages(messages, chatMessagesElement) {
    if (!chatMessagesElement) return;
    chatMessagesElement.innerHTML = '';
    messages.forEach(msg => { if (msg.role !== 'system') renderMessage(msg, chatMessagesElement, msg.role === 'user'); });
    if (chatMessagesElement.firstChild) chatMessagesElement.scrollTop = chatMessagesElement.scrollHeight;
}
export function addChatMessage(message, chatMessagesElement, isUserMessage) {
    if (message.role === 'system') return;
    renderMessage(message, chatMessagesElement, isUserMessage);
}

export function updateStreamingMessage(fullMessageContent, chatMessagesElement) {
    if (!chatMessagesElement) return;
    let assistantMessageGroup = chatMessagesElement.querySelector('.message-group.bot.streaming:last-child');
    let messageBubble, contentWrapper;
    if (!assistantMessageGroup) {
        assistantMessageGroup = document.createElement('div'); assistantMessageGroup.className = 'message-group bot streaming';
        const avatar = document.createElement('div'); avatar.className = 'bot-avatar'; avatar.textContent = 'B';
        const messagesContainer = document.createElement('div'); messagesContainer.className = 'messages';
        messageBubble = document.createElement('div'); messageBubble.className = 'chat-message bot';
        contentWrapper = document.createElement('div'); messageBubble.appendChild(contentWrapper);
        messagesContainer.appendChild(messageBubble);
        assistantMessageGroup.appendChild(avatar); assistantMessageGroup.appendChild(messagesContainer);
        chatMessagesElement.appendChild(assistantMessageGroup);
    } else {
        messageBubble = assistantMessageGroup.querySelector('.chat-message.bot');
        contentWrapper = messageBubble.firstChild;
        if (!contentWrapper || contentWrapper.nodeName === '#text' || contentWrapper.classList.contains('user-images')) {
            contentWrapper = document.createElement('div');
            const existingImages = messageBubble.querySelector('.user-images');
            messageBubble.innerHTML = ''; 
            if(existingImages) messageBubble.appendChild(existingImages);
            messageBubble.appendChild(contentWrapper);
        }
    }
    if (typeof marked !== 'undefined') {
        contentWrapper.innerHTML = marked.parse(fullMessageContent);
        processAndAddCodeBlockFunctionality(contentWrapper);
    } else { contentWrapper.textContent = fullMessageContent; }
    chatMessagesElement.scrollTop = chatMessagesElement.scrollHeight;
}
export function finalizeStreamingMessage(chatMessagesElement) {
    const streamingGroup = chatMessagesElement.querySelector('.message-group.bot.streaming:last-child');
    if (streamingGroup) streamingGroup.classList.remove('streaming');
}

export function addErrorMessageWithRetry(errorMessage, chatMessagesElement, retryCallback) {
    if (errorMessage.role === 'system' || !chatMessagesElement) return;
    const messageGroup = document.createElement('div'); messageGroup.className = 'message-group bot error-message-group';
    const avatar = document.createElement('div'); avatar.className = 'bot-avatar'; avatar.style.backgroundColor = 'var(--accent-danger)'; avatar.textContent = '!';
    const messagesContainer = document.createElement('div'); messagesContainer.className = 'messages';
    const messageBubble = document.createElement('div'); messageBubble.className = 'chat-message bot';
    messageBubble.style.backgroundColor = 'var(--accent-danger)'; messageBubble.style.color = 'white';
    const contentP = document.createElement('p'); contentP.textContent = errorMessage.content || 'An unknown error occurred.';
    messageBubble.appendChild(contentP);
    if (retryCallback) {
        const retryButton = document.createElement('button'); retryButton.textContent = 'Retry';
        retryButton.className = 'button-secondary';
        Object.assign(retryButton.style, { borderColor: 'white', color: 'white', marginLeft: '10px', padding: '4px 8px', fontSize: '0.8em', marginTop: '8px' });
        retryButton.onclick = () => { messageGroup.remove(); retryCallback(); };
        messageBubble.appendChild(retryButton);
    }
    messagesContainer.appendChild(messageBubble);
    messageGroup.appendChild(avatar); messageGroup.appendChild(messagesContainer);
    chatMessagesElement.appendChild(messageGroup);
    chatMessagesElement.scrollTop = chatMessagesElement.scrollHeight;
}

// Store handlers at module level to ensure they can be removed.
let sendButtonMouseOverHandler = null;
let sendButtonMouseOutHandler = null;

export function showLoadingIndicator(isLoading, buttonElement = null, loadingText = "Loading...", isAbortable = false) {
    const overlay = document.getElementById('loading-overlay');
    const targetButton = buttonElement || document.getElementById('send-button');

    if (isLoading) {
        if (targetButton) {
            if (!targetButton.dataset.originalHTML) { // First time setting loading state for this cycle
                targetButton.dataset.originalHTML = targetButton.innerHTML;
                targetButton.dataset.originalBG = targetButton.style.backgroundColor || ""; // Store original BG or empty if none
            }
            targetButton.disabled = false; // Keep button enabled for hover/click if abortable
            targetButton.innerHTML = `<div class="spinner spinner-button"></div><span style="vertical-align: middle;">${loadingText}</span>`;
            
            // Set non-stop default style (e.g. original or a specific "sending" color if different)
            // For now, assume it re-uses originalBG unless specifically styled for "sending"
            targetButton.style.backgroundColor = targetButton.dataset.originalBG; 

            if (isAbortable) {
                const currentLoadingText = loadingText; // Capture for closure

                sendButtonMouseOverHandler = () => {
                    if (targetButton.dataset.isLoading === 'true') { // Still in loading state
                        targetButton.innerHTML = `<div class="spinner spinner-button"></div><span style="vertical-align: middle;">Stop</span>`;
                        targetButton.style.backgroundColor = 'var(--accent-danger)';
                        targetButton.dataset.canAbort = 'true';
                    }
                };
                sendButtonMouseOutHandler = () => {
                    // Only revert if still loading and was in 'canAbort' state (mouse was over)
                    if (targetButton.dataset.isLoading === 'true' && targetButton.dataset.canAbort === 'true') { 
                        targetButton.innerHTML = `<div class="spinner spinner-button"></div><span style="vertical-align: middle;">${currentLoadingText}</span>`;
                        targetButton.style.backgroundColor = targetButton.dataset.originalBG; // Revert to original/sending BG
                        targetButton.dataset.canAbort = 'false';
                    }
                };
                targetButton.addEventListener('mouseenter', sendButtonMouseOverHandler);
                targetButton.addEventListener('mouseleave', sendButtonMouseOutHandler);
            }
            targetButton.dataset.isLoading = 'true'; // Mark as loading
        } else if (overlay) { // Global overlay
            overlay.classList.remove('hidden');
        }
    } else { // isLoading is false
        if (targetButton) {
            targetButton.disabled = false;
            if (targetButton.dataset.originalHTML) {
                targetButton.innerHTML = targetButton.dataset.originalHTML;
            } else {
                targetButton.textContent = "Send"; // Default text for send button
            }
            if (targetButton.dataset.originalBG !== undefined) {
                 targetButton.style.backgroundColor = targetButton.dataset.originalBG;
            }
            
            // Clean up event listeners and data attributes
            if (sendButtonMouseOverHandler) {
                targetButton.removeEventListener('mouseenter', sendButtonMouseOverHandler);
                sendButtonMouseOverHandler = null;
            }
            if (sendButtonMouseOutHandler) {
                targetButton.removeEventListener('mouseleave', sendButtonMouseOutHandler);
                sendButtonMouseOutHandler = null;
            }
            delete targetButton.dataset.originalHTML;
            delete targetButton.dataset.originalBG;
            delete targetButton.dataset.canAbort;
            delete targetButton.dataset.isLoading; 
        }
        if (overlay) overlay.classList.add('hidden');
    }
}


export function updateActionButtonsVisibility(currentConversationId, currentChatMessages) {
    const forkBtn = document.getElementById('fork-conversation-btn');
    const exportBtn = document.getElementById('export-chat-btn');
    const displayableMessagesLength = currentChatMessages.filter(m => m.role !== 'system').length;

    if (forkBtn) {
        if (currentConversationId && displayableMessagesLength > 0) forkBtn.classList.remove('hidden');
        else forkBtn.classList.add('hidden');
    }
    if (exportBtn) {
        if (displayableMessagesLength > 0) exportBtn.classList.remove('hidden');
        else exportBtn.classList.add('hidden');
    }
}


export function populateForkModal(currentChatMessages, forkMessageListElement, selectionHandler) {
    if (!forkMessageListElement) return;
    forkMessageListElement.innerHTML = '';
    const displayableMessages = currentChatMessages.filter(msg => msg.role !== 'system');
    if (displayableMessages.length === 0) {
        forkMessageListElement.innerHTML = '<li style="padding: 10px; color: var(--text-secondary);">No messages to fork.</li>'; return;
    }
    displayableMessages.forEach((msg) => {
        const originalIndex = currentChatMessages.indexOf(msg);
        const listItem = document.createElement('li');
        let preview = msg.content ? msg.content.substring(0, 70) : "";
        if (msg.content && msg.content.length > 70) preview += "...";
        let imageIndicator = (msg.images && msg.images.length > 0) ? `[${msg.images.length} Image(s)] ` : "";
        if (!preview.trim() && imageIndicator) preview = "[Image content]"; else if (!preview.trim()) preview = "[Empty message]";
        listItem.textContent = `${msg.role === 'user' ? 'You' : 'Bot'}: ${imageIndicator}${preview}`;
        listItem.dataset.messageIndex = originalIndex;
        listItem.addEventListener('click', selectionHandler);
        forkMessageListElement.appendChild(listItem);
    });
}

export function setupSidebarToggle(toggleButton, sidebar) {
    if (!toggleButton || !sidebar) return;
    const checkScreenWidth = () => {
        if (window.innerWidth <= 768) {
            toggleButton.style.display = 'flex';
            if (!sidebar.classList.contains('open')) sidebar.style.left = '-280px';
        } else {
            toggleButton.style.display = 'none'; sidebar.style.left = '0'; sidebar.classList.remove('open');
        }
    };
    checkScreenWidth(); window.addEventListener('resize', checkScreenWidth);
    toggleButton.addEventListener('click', (e) => {
        e.stopPropagation(); sidebar.classList.toggle('open');
        sidebar.style.left = sidebar.classList.contains('open') ? '0' : '-280px';
    });
    document.addEventListener('click', (event) => {
        if (window.innerWidth <= 768 && sidebar.classList.contains('open')) {
            if (!sidebar.contains(event.target) && !toggleButton.contains(event.target)) {
                sidebar.classList.remove('open'); sidebar.style.left = '-280px';
            }
        }
    });
}

export function handleImagePreviews(imageInputElement, imagePreviewsContainer) {
    if (!imageInputElement || !imagePreviewsContainer) {
        return { getFiles: () => [], clear: () => {}, addFiles: () => {} };
    }
    let currentFiles = [];

    const cleanInput = imageInputElement.cloneNode(true);
    imageInputElement.parentNode.replaceChild(cleanInput, imageInputElement);
    imageInputElement = cleanInput;

    function renderPreviews() {
        imagePreviewsContainer.innerHTML = '';
        currentFiles.forEach((file, index) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const previewWrapper = document.createElement('div'); previewWrapper.className = 'image-preview-item';
                const img = document.createElement('img'); img.src = e.target.result; previewWrapper.appendChild(img);
                const removeBtn = document.createElement('button'); removeBtn.type = 'button'; removeBtn.innerHTML = '×';
                removeBtn.className = 'remove-image-btn';
                removeBtn.onclick = () => {
                    currentFiles.splice(index, 1);
                    const dataTransfer = new DataTransfer();
                    currentFiles.forEach(f => dataTransfer.items.add(f));
                    imageInputElement.files = dataTransfer.files;
                    renderPreviews();
                };
                previewWrapper.appendChild(removeBtn); imagePreviewsContainer.appendChild(previewWrapper);
            };
            reader.readAsDataURL(file);
        });
    }

    imageInputElement.addEventListener('change', (event) => {
        currentFiles = Array.from(event.target.files);
        renderPreviews();
    });

    const clearPreviewsAndFiles = () => {
        currentFiles = []; imagePreviewsContainer.innerHTML = ''; imageInputElement.value = '';
    };

    const addFilesProgrammatically = (filesToAdd) => {
        const newFileArray = [...currentFiles];
        for (const file of filesToAdd) {
            if (file instanceof File && !newFileArray.some(f => f.name === file.name && f.lastModified === file.lastModified && f.size === file.size)) {
                newFileArray.push(file);
            }
        }
        currentFiles = newFileArray;
        const dataTransfer = new DataTransfer();
        currentFiles.forEach(f => dataTransfer.items.add(f));
        imageInputElement.files = dataTransfer.files;
        renderPreviews();
    };

    return { getFiles: () => currentFiles, clear: clearPreviewsAndFiles, addFiles: addFilesProgrammatically };
}