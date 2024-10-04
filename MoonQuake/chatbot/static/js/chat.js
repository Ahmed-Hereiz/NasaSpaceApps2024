
$(document).ready(function() {
    const chatContainer = $('#chatContainer');
    const chatForm = $('#chatForm');
    const userInput = $('#userInput');
    const chatbotUrl = chatContainer.attr('chatbot-url');
    const csrfToken = $('[name=csrfmiddlewaretoken]').val();

    function addMessage(message, isUser = false) {
        const messageHtml = `
            <li class="chat ${isUser ? 'user' : 'bot'}">
                <div class="message">${message}</div>
            </li>
        `;
        chatContainer.append(messageHtml);
        chatContainer.scrollTop(chatContainer[0].scrollHeight);
    }

    chatForm.on('submit', function(e) {
        e.preventDefault();
        
        const message = userInput.val().trim();
        if (!message) return;
        addMessage(message, true);
        userInput.val('');
        $.ajax({
            url: chatbotUrl, 
            type: 'POST',
            contentType: 'application/json',
            headers: {
                'X-CSRFToken': csrfToken
            },
            data: JSON.stringify({
                message: message
            }),
            success: function (response) {
                console.log(response.message)
                addMessage(response.message);
            },
            error: function() {
                addMessage("Sorry, there was an error processing your request.", false);
            }
        });
    });

    userInput.on('keydown', function(e) {
        if (e.keyCode === 13 && !e.shiftKey) {
            e.preventDefault();
            chatForm.submit();
        }
    });
});