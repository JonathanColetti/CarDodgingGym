document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('gameCanvas');
    const ctx = canvas.getContext('2d');
    const startButton = document.getElementById('startButton');
    const scoreDisplay = document.getElementById('scoreDisplay');

    const game = new CarGame(ctx, canvas.width, canvas.height);
    let ortSession;
    let gameActive = false;
    let currentObs;
    let normalizer;

    /*
    */
    async function setupGame() {
        try {
            // Load the ONNX model
            ortSession = await ort.InferenceSession.create('./ppo_cargame.onnx');
        
            
            // Load VecNormalize statistics
            const statsResponse = await fetch('./vecnormalize_stats.json');
            if (!statsResponse.ok) {
                throw new Error('Failed to load normalization stats. Run the stats export script first!');
            }
            const stats = await statsResponse.json();
            normalizer = new ObservationNormalizer(stats);
            
            statusDisplay.textContent = 'Loading game assets...';
            await game.loadAssets();
            
            startButton.disabled = false;
            startButton.textContent = 'Start Game';
            statusDisplay.textContent = 'Ready to play!';
            statusDisplay.className = 'success';
            
            currentObs = game.reset();
            game.render();
            scoreDisplay.textContent = 'Score: 0';
            
            console.log('Normalization stats loaded:', {
                mean: normalizer.mean,
                var: normalizer.var,
                epsilon: normalizer.epsilon,
                clip_obs: normalizer.clip_obs
            });
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
        if (!gameActive) {
            console.error("game is not actice");
            return;
        }
        if (!normalizer) {
            console.error("normalizer is null");
            return;
        }
        const normalizedObs = normalizer.normalize(currentObs);
        const obsFloat32 = new Float32Array(normalizedObs);
        const obsTensor = new ort.Tensor('float32', obsFloat32, [1, 3]);
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