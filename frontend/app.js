class RAGFrontend {
    constructor() {
        this.baseUrl = window.location.origin;
        this.init();
    }

    async init() {
        this.bindEvents();
        await this.startupCheck();
        await this.loadHistory();
    }

    bindEvents() {
        document.getElementById('askBtn').addEventListener('click', () => this.askQuestion());
        document.getElementById('question').addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.askQuestion();
            }
        });

        document.getElementById('reindexBtn').addEventListener('click', () => this.reindex());
        document.getElementById('configBtn').addEventListener('click', () => this.showConfig());
        document.getElementById('clearHistoryBtn').addEventListener('click', () => this.clearHistory());
        document.getElementById('saveConfigBtn').addEventListener('click', () => this.saveConfig());
        document.getElementById('closeConfigBtn').addEventListener('click', () => this.hideConfig());
    }

    async startupCheck() {
        try {
            const response = await fetch('/startup');
            const data = await response.json();
            
            if (data.needs_update) {
                this.showStatus(
                    `Changes detected: ${data.added.length} added, ${data.removed.length} removed, ${data.modified.length} modified. Reindex recommended.`,
                    'warning'
                );
            }
        } catch (error) {
            console.error('Startup check failed:', error);
        }
    }

    async askQuestion() {
        const questionInput = document.getElementById('question');
        const question = questionInput.value.trim();
        
        if (!question) return;

        // Add user message to chat
        this.addMessage(question, 'user');
        questionInput.value = '';

        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question })
            });

            const data = await response.json();
            
            // Add assistant message
            this.addMessage(data.answer, 'assistant');
            
            // Show context
            this.showContext(data.context_used);
            
        } catch (error) {
            console.error('Error asking question:', error);
            this.addMessage('Error: Could not get response from server', 'assistant');
        }
    }

    addMessage(content, role) {
        const chat = document.getElementById('chat');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const timestamp = new Date().toLocaleTimeString();
        messageDiv.innerHTML = `
            <div class="timestamp">${timestamp}</div>
            <div class="content">${role === 'assistant' ? marked.parse(content) : content}</div>
        `;
        
        chat.appendChild(messageDiv);
        chat.scrollTop = chat.scrollHeight;
    }

    showContext(contextItems) {
        const panel = document.getElementById('contextPanel');
        const list = document.getElementById('contextList');
        
        if (!contextItems || contextItems.length === 0) {
            panel.style.display = 'none';
            return;
        }

        list.innerHTML = contextItems.map(item => `
            <div class="context-item">
                <h4>${item.file} (lines ${item.start}-${item.end})</h4>
                <div class="preview">${this.escapeHtml(item.preview)}</div>
            </div>
        `).join('');

        panel.style.display = 'block';
    }

    async reindex() {
        this.showStatus('Reindexing... This may take a while.', 'info');
        
        try {
            const response = await fetch('/reindex', { method: 'POST' });
            const data = await response.json();
            this.showStatus(data.message, 'info');
        } catch (error) {
            console.error('Reindex failed:', error);
            this.showStatus('Reindex failed', 'warning');
        }
    }

    async showConfig() {
        try {
            const response = await fetch('/config');
            const data = await response.json();
            document.getElementById('configFiles').value = data.files.join('\n');
            document.getElementById('configModal').style.display = 'block';
        } catch (error) {
            console.error('Failed to load config:', error);
        }
    }

    hideConfig() {
        document.getElementById('configModal').style.display = 'none';
    }

    async saveConfig() {
        const filesText = document.getElementById('configFiles').value;
        const files = filesText.split('\n').filter(f => f.trim());
        
        try {
            await fetch('/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ files })
            });
            
            this.hideConfig();
            this.showStatus('Configuration updated', 'info');
            await this.startupCheck();
        } catch (error) {
            console.error('Failed to save config:', error);
        }
    }

    async clearHistory() {
        if (!confirm('Clear all chat history?')) return;
        
        try {
            await fetch('/history', { method: 'DELETE' });
            document.getElementById('chat').innerHTML = '';
            this.showStatus('History cleared', 'info');
        } catch (error) {
            console.error('Failed to clear history:', error);
        }
    }

    async loadHistory() {
        try {
            const response = await fetch('/history');
            const history = await response.json();
            
            history.reverse().forEach(item => {
                this.addMessage(item.question, 'user');
                this.addMessage(item.answer, 'assistant');
            });
        } catch (error) {
            console.error('Failed to load history:', error);
        }
    }

    showStatus(message, type) {
        const status = document.getElementById('status');
        status.textContent = message;
        status.className = `status ${type}`;
        status.style.display = 'block';
        
        setTimeout(() => {
            status.style.display = 'none';
        }, 5000);
    }

    escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    new RAGFrontend();
});