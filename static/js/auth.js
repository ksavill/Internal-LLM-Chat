// static/js/auth.js
import { TOKEN_KEY } from './config.js';
import * as api from './api.js';
import * as ui from './ui.js';
import * as chat from './chat.js';

export function getToken() {
    return localStorage.getItem(TOKEN_KEY);
}

function setToken(token) {
    localStorage.setItem(TOKEN_KEY, token);
}

function clearToken() {
    localStorage.removeItem(TOKEN_KEY);
}

export function clearTokenAndReload() {
    clearToken();
    window.location.hash = ''; // Clear hash before reload
    window.location.reload();
}

export function initAuth(elements) {
    const {
        loginView, chatView, loginForm, signupForm, logoutButton,
        authTitle, loginPrompt, signupPrompt, toggleAuthFormButton,
        loginError, signupError, signupMessage,
    } = elements;

    const updateAuthView = (isLoggedIn) => {
        if (isLoggedIn) {
            ui.hideElement(loginView); // Hides modal by removing 'active'
            ui.showElement(chatView, 'grid'); // Show chat app container
            chat.loadInitialChatData();
        } else {
            ui.showElement(loginView); // Shows modal by adding 'active'
            ui.hideElement(chatView);
            authTitle.textContent = 'Login';
            ui.showElement(loginForm);
            ui.hideElement(signupForm);
            ui.showElement(loginPrompt);
            ui.hideElement(signupPrompt);
            toggleAuthFormButton.textContent = 'Sign Up';
        }
    };
    
    if (getToken()) {
        updateAuthView(true);
    } else {
        updateAuthView(false);
    }

    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        ui.clearMessages([loginError, signupMessage, signupError]);
        const username = loginForm.username.value;
        const password = loginForm.password.value;
        const loginButton = loginForm.querySelector('button[type="submit"]');
        ui.showLoadingIndicator(true, loginButton, "Logging in...");
        try {
            const data = await api.loginUser(username, password);
            setToken(data.token);
            updateAuthView(true);
            loginForm.reset();
        } catch (error) {
            ui.displayAuthError(loginError, error.message);
        } finally {
            ui.showLoadingIndicator(false, loginButton, "Login");
        }
    });

    signupForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        ui.clearMessages([loginError, signupMessage, signupError]);
        const username = signupForm.username.value;
        const password = signupForm.password.value;
        const signupButton = signupForm.querySelector('button[type="submit"]');
        ui.showLoadingIndicator(true, signupButton, "Signing up...");
        try {
            const data = await api.signupUser(username, password);
            ui.displayAuthMessage(signupMessage, data.message + " Please log in.");
            authTitle.textContent = 'Login';
            ui.showElement(loginForm); ui.hideElement(signupForm);
            ui.showElement(loginPrompt); ui.hideElement(signupPrompt);
            toggleAuthFormButton.textContent = 'Sign Up';
            signupForm.reset();
        } catch (error) {
            ui.displayAuthError(signupError, error.message);
        } finally {
            ui.showLoadingIndicator(false, signupButton, "Sign Up");
        }
    });

    logoutButton.addEventListener('click', () => {
        clearToken();
        chat.resetChatState();
        updateAuthView(false);
    });

    toggleAuthFormButton.addEventListener('click', () => {
        ui.clearMessages([loginError, signupMessage, signupError]);
        if (loginForm.classList.contains('hidden')) {
            authTitle.textContent = 'Login';
            ui.showElement(loginForm); ui.hideElement(signupForm);
            ui.showElement(loginPrompt); ui.hideElement(signupPrompt);
            toggleAuthFormButton.textContent = 'Sign Up';
        } else {
            authTitle.textContent = 'Sign Up';
            ui.hideElement(loginForm); ui.showElement(signupForm);
            ui.hideElement(loginPrompt); ui.showElement(signupPrompt);
            toggleAuthFormButton.textContent = 'Login';
        }
    });
}