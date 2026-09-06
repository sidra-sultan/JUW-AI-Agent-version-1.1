
const form = document.querySelector('#form');
const input = document.querySelector('#message');
const chat = document.querySelector('#chat');
const clearButton = document.querySelector('#clear-chat');

const history = [];

const sessionId =
    window.crypto?.randomUUID?.() ||
    `${Date.now()}-${Math.random().toString(16).slice(2)}`;

// Keep only the most recent exchanges in the browser.
// This mirrors the backend limit so long conversations
// do not eventually exceed the API's accepted history size.
const MAX_HISTORY_ITEMS = 40;


function addMessage(text, role) {
    const message = document.createElement('article');
    message.className = role;

    if (role === 'assistant') {
        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.textContent = 'J';

        const content = document.createElement('div');

        text
            .split('\n')
            .filter(Boolean)
            .forEach(line => {
                const p = document.createElement('p');
                p.textContent = line;
                content.append(p);
            });

        message.append(avatar, content);
    } else {
        message.textContent = text;
    }

    chat.append(message);

    message.scrollIntoView({
        behavior: 'smooth',
        block: 'end'
    });

    return message;
}


function addTypingIndicator() {
    const message = document.createElement('article');
    message.className = 'assistant typing';

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = 'J';

    const content = document.createElement('div');

    const p = document.createElement('p');
    p.textContent = '…';

    content.append(p);
    message.append(avatar, content);
    chat.append(message);

    message.scrollIntoView({
        behavior: 'smooth',
        block: 'end'
    });

    return message;
}


// Convert whatever error shape the server returns
// into a readable message.
function describeError(data, res) {
    const detail = data && data.detail;

    if (Array.isArray(detail)) {
        const parts = detail
            .map(item =>
                item && typeof item === 'object'
                    ? item.msg
                    : item
            )
            .filter(
                part =>
                    typeof part === 'string' &&
                    part.trim()
            );

        if (parts.length) {
            return parts.join('; ');
        }
    } else if (
        typeof detail === 'string' &&
        detail.trim()
    ) {
        return detail;
    } else if (
        detail &&
        typeof detail === 'object'
    ) {
        if (
            typeof detail.message === 'string'
        ) {
            return detail.message;
        }

        if (
            typeof detail.msg === 'string'
        ) {
            return detail.msg;
        }
    }

    if (res) {
        if (res.status === 429) {
            return (
                'The AI service is temporarily rate-limited.' + 
                ' Please try again in a moment.'
            );
        }

        if (res.status >= 500) {
            return (
                'The assistant ran into a problem ' +
                'answering that. Please try again ' +
                'in a moment.'
            );
        }
    }

    return 'Something went wrong. Please try again.';
}


async function sendMessage(message) {
    if (!message) {
        return;
    }

    addMessage(message, 'user');

    input.value = '';

    const button = form.querySelector('button');
    button.disabled = true;

    const typingIndicator = addTypingIndicator();

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message,
                history,
                session_id: sessionId
            })
        });

        let data = {};

        try {
            data = await res.json();
        } catch {
            // Response was not valid JSON.
            data = {};
        }

        if (!res.ok) {
            throw new Error(
                describeError(data, res)
            );
        }

        if (
            typeof data.answer !== 'string'
        ) {
            throw new Error(
                'Received an unexpected response from the server.'
            );
        }

        typingIndicator.remove();

        // Only display the clean AI answer.
        //
        // The backend may still return `sources`,
        // but they are intentionally NOT rendered
        // in the user interface.
        addMessage(
            data.answer,
            'assistant'
        );

        history.push(
            {
                role: 'user',
                content: message
            },
            {
                role: 'assistant',
                content: data.answer
            }
        );

        if (
            history.length >
            MAX_HISTORY_ITEMS
        ) {
            history.splice(
                0,
                history.length -
                MAX_HISTORY_ITEMS
            );
        }

    } catch (error) {
        typingIndicator.remove();

        addMessage(
            error.message ||
            'Something went wrong. Please try again.',
            'assistant'
        );

    } finally {
        button.disabled = false;
        input.focus();
    }
}


form.addEventListener(
    'submit',
    event => {
        event.preventDefault();
        sendMessage(
            input.value.trim()
        );
    }
);


document
    .querySelectorAll('[data-question]')
    .forEach(button => {

        button.addEventListener(
            'click',
            () => {
                sendMessage(
                    button.dataset.question
                );
            }
        );

    });


clearButton.addEventListener(
    'click',
    () => {

        history.length = 0;

        chat.innerHTML = '';

        addMessage(
            'Assalam o Alaikum! I’m the JUW AI Assistant.\n' +
            'I can help with information available in the official JUW knowledge base.',
            'assistant'
        );

        input.focus();
    }
);

