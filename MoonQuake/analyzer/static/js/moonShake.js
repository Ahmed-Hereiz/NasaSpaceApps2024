function shakeLogo() {
    const logo = document.querySelector('.logo');
    const logoh1 = document.querySelector('.logoh1');
    logo.classList.add('shake');
    logoh1.classList.add('shakeColor');
    
    setTimeout(() => {
        logo.classList.remove('shake');
        logoh1.classList.remove('shakeColor');
    }, 2000); // 500 milliseconds = 0.5 seconds
}

// Initial shake
shakeLogo();

// Repeat every 3 minutes
setInterval(shakeLogo, 0.5 * 60 * 1000); // 3 minutes in milliseconds