// static/js/main.js
import * as auth from './auth.js';
import * as ui from './ui.js';
import * as chat from './chat.js';

document.addEventListener('DOMContentLoaded', () => {
    const elements = {
        loginView: document.getElementById('login-view'),
        chatView: document.getElementById('chat-view'),
        loginForm: document.getElementById('login-form'),
        signupForm: document.getElementById('signup-form'),
        logoutButton: document.getElementById('logout-button'),
        authTitle: document.getElementById('auth-title'),
        loginPrompt: document.getElementById('login-prompt'),
        signupPrompt: document.getElementById('signup-prompt'),
        toggleAuthFormButton: document.getElementById('toggle-auth-form-button'),
        loginError: document.getElementById('login-error'),
        signupError: document.getElementById('signup-error'),
        signupMessage: document.getElementById('signup-message'),
        themeToggleButtons: [document.getElementById('theme-toggle-auth'), document.getElementById('theme-toggle-chat')],
        chatMessagesElement: document.getElementById('chat-messages'),
        conversationsListElement: document.getElementById('conversations-list'),
        messageInputElement: document.getElementById('user-input'),
        modelSelectElement: document.getElementById('model-select'),
        backupModelSelectElement: document.getElementById('backup-model-select'), // Changed ID
        profileSelectElement: document.getElementById('profile-select'),
        streamToggleElement: document.getElementById('stream-toggle'),
        imageInputElement: document.getElementById('image-input'),
        imagePreviewsContainer: document.getElementById('image-previews'),
        conversationTitleElement: document.getElementById('conversation-title'),
        chatFormElement: document.getElementById('chat-form'),
        sendButton: document.getElementById('send-button'),
        newChatButton: document.getElementById('new-chat-button'),
        forkConversationBtnElement: document.getElementById('fork-conversation-btn'),
        exportChatButton: document.getElementById('export-chat-btn'),
        forkModalElement: document.getElementById('fork-modal'),
        cancelForkBtnElement: document.getElementById('cancel-fork-btn'),
        forkMessageListElement: document.getElementById('fork-message-list'),
        sidebarToggle: document.getElementById('sidebar-toggle'),
        sidebar: document.getElementById('sidebar'),
    };

    ui.setupThemeToggle(elements.themeToggleButtons);
    auth.initAuth(elements); 
    chat.initChatElements(elements);
    
    if (elements.newChatButton) {
        elements.newChatButton.addEventListener('click', () => chat.startNewChat(true));
    }
    if (elements.exportChatButton) {
        elements.exportChatButton.addEventListener('click', chat.exportChat);
    }
    
    ui.setupSidebarToggle(elements.sidebarToggle, elements.sidebar);

    window.addEventListener('hashchange', () => {
        if (auth.getToken()) {
            const hash = window.location.hash.substring(1);
            if (hash) chat.handleConversationSelection(hash);
            else if (!hash && auth.getToken()) chat.startNewChat(true);
        }
    });
});