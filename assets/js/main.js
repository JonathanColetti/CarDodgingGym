document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('gameCanvas');
    const ctx = canvas.getContext('2d');
    const startButton = document.getElementById('startButton');
    const scoreDisplay = document.getElementById('scoreDisplay');

    const game = new CarGame(ctx, canvas.width, canvas.height);
    let ortSession;
    let gameActive = false;
    let currentObs;

    /*
    */
    async function setupGame() {
        try {
            // load the ONNX model
            ortSession = await ort.InferenceSession.create('./ppo_cargame.onnx');
            
            await game.loadAssets();

            startButton.disabled = false;
            startButton.textContent = 'Start Game';
            
            currentObs = game.reset();
            game.render();
            scoreDisplay.textContent = 'Score: 0';

        } catch (error) {
            console.error("Failed to load game:", error);
            startButton.textContent = 'Failed to Load';
        }
    }

    /*
    */
   function startGame() {
        // stop the chance of multiple starts
        if (gameActive) return; 
        
        currentObs = game.reset();
        gameActive = true;
        startButton.disabled = true;
        startButton.textContent = 'Playing...';
        
        // Start the game loop
        requestAnimationFrame(gameLoop);
    }
    
    /*

    */
    async function gameLoop() {
        if (!gameActive) return;

        const obsFloat32 = new Float32Array(currentObs);
        const obsTensor = new ort.Tensor('float32', obsFloat32, [1, 4]);
        const inputs = { 'obs': obsTensor };

        const results = await ortSession.run(inputs);
        const action = Number(results.action.data[0]);

        const { observation, terminated, info } = game.step(action);
        currentObs = observation;

        game.render();

        scoreDisplay.textContent = `Score: ${info.score}`;

        if (terminated) {
            gameActive = false;
            startButton.disabled = false;
            startButton.textContent = 'Play Again?';
            console.log(`Game Over! Final Score: ${info.score}`);
        } else {
            requestAnimationFrame(gameLoop);
        }
    }

    startButton.addEventListener('click', startGame);

    setupGame();
});