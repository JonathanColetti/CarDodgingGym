class CarGame {
    constructor(ctx, width, height) {
        this.ctx = ctx;
        this.SCREEN_WIDTH = width;
        this.SCREEN_HEIGHT = height;

        // consts from Python
        this.REWARD_SUCCESSFUL_DODGE = 5.0;
        this.REWARD_ALIVE_PER_STEP = 0.5;
        this.PENALTY_LANE_CHANGE = -0.05;
        this.PENALTY_CRASH = -5.0;

        this.GRASS_COLOR = "rgb(60, 220, 0)";
        this.DARK_ROAD_COLOR = "rgb(50, 50, 50)";
        this.YELLOW_LINE_COLOR = "rgb(255, 240, 60)";
        this.WHITE_LINE_COLOR = "rgb(255, 255, 255)";

        this.road_w = Math.floor(this.SCREEN_WIDTH / 1.6);
        this.roadmark_w = Math.floor(this.SCREEN_WIDTH / 80);
        this.right_lane = this.SCREEN_WIDTH / 2 + this.road_w / 4;
        this.left_lane = this.SCREEN_WIDTH / 2 - this.road_w / 4;
        this.initial_speed = 3;

        // loaded in loadAssets
        this.carImg = new Image();
        this.car2Img = new Image();
        this.carWidth = 0;
        this.carHeight = 0;
        this.car2Width = 0;
        this.car2Height = 0;

        this.speed = 0;
        this.score = 0;
        this.level = 0;
        this.line_offset = 0;
        // x, y, width, height
        this.car_loc = {}; 
        this.car2_loc = {};
    }
    /*
    */
    async loadAssets() {
        const loadImg = (img, src) => {
            return new Promise((resolve, reject) => {
                img.onload = () => resolve(img);
                img.onerror = reject;
                img.src = src;
            });
        };

        await Promise.all([
            loadImg(this.carImg, "assets/cars/car.png"),
            loadImg(this.car2Img, "assets/cars/otherCar.png")
        ]);

        // Fake pygame.transform.scale (roughly .8x)
        this.carWidth = this.carImg.width * 0.8;
        this.carHeight = this.carImg.height * 0.8;
        this.car2Width = this.car2Img.width * 0.8;
        this.car2Height = this.car2Img.height * 0.8;
    }

    /*
    */
    _get_obs() {
        // player_lane_obs = 0 if self.car_loc.centerx == self.left_lane else 1
        const player_lane_obs = (this.car_loc.x + this.car_loc.width / 2) === this.left_lane ? 0 : 1;
        const enemy_lane_obs = (this.car2_loc.x + this.car2_loc.width / 2) === this.left_lane ? 0 : 1;
        
        // enemy_y_norm = self.car2_loc.center[1] / self.SCREEN_HEIGHT
        const enemy_y_center = this.car2_loc.y + this.car2_loc.height / 2;
        const enemy_y_norm = enemy_y_center / this.SCREEN_HEIGHT;

        return [player_lane_obs, enemy_lane_obs, enemy_y_norm];
    }

    /*
    */
    _get_info() {
        return { score: this.score, level: this.level };
    }

    /*
    */
    reset() {
        this.speed = this.initial_speed;
        this.score = 0;
        this.level = 0;
        this.line_offset = 0;

        // ppos car position
        const player_start_lane = Math.floor(Math.random() * 2) === 0 ? this.left_lane : this.right_lane;
        this.car_loc = {
            x: player_start_lane - this.carWidth / 2,
            y: this.SCREEN_HEIGHT * 0.85 - this.carHeight / 2,
            width: this.carWidth,
            height: this.carHeight
        };

        // static car position
        const enemy_start_lane = Math.floor(Math.random() * 2) === 0 ? this.left_lane : this.right_lane;
        this.car2_loc = {
            x: enemy_start_lane - this.car2Width / 2,
            y: -this.car2Height,
            width: this.car2Width,
            height: this.car2Height
        };

        return this._get_obs();
    }

    /*
    */
    _colliderect(rect1, rect2) {
        return (
            rect1.x < rect2.x + rect2.width &&
            rect1.x + rect1.width > rect2.x &&
            rect1.y < rect2.y + rect2.height &&
            rect1.y + rect1.height > rect2.y
        );
    }

    /*
    */
    step(action) {
        let reward = 0;
        const current_lane_center = this.car_loc.x + this.car_loc.width / 2;

        // (move left)
        if (action === 0 && current_lane_center === this.right_lane) {
            this.car_loc.x = this.left_lane - this.carWidth / 2;
            reward += this.PENALTY_LANE_CHANGE;
        }
        // (move right)
        else if (action === 2 && current_lane_center === this.left_lane) {
            this.car_loc.x = this.right_lane - this.carWidth / 2;
            reward += this.PENALTY_LANE_CHANGE;
        }
        // (do nothing)
        this.car2_loc.y += this.speed;

        // next level
        if (this.score > 0 && this.score % 5 === 0 && this.level < this.score) {
            this.speed += 0.5;
            this.level += 1;
        }

        if (this.car2_loc.y > this.SCREEN_HEIGHT) {
            reward += this.REWARD_SUCCESSFUL_DODGE;
            this.score += 1;
            
            const enemy_start_lane = Math.floor(Math.random() * 2) === 0 ? this.right_lane : this.left_lane;
            this.car2_loc = {
                x: enemy_start_lane - this.car2Width / 2,
                y: -this.car2Height,
                width: this.car2Width,
                height: this.car2Height
            };
        }
        

        // collision
        const terminated = this._colliderect(this.car_loc, this.car2_loc);
        if (terminated) {
            reward += this.PENALTY_CRASH;
        } else {
            reward += this.REWARD_ALIVE_PER_STEP;
        }

        const observation = this._get_obs();
        const info = this._get_info();

        return { observation, reward, terminated, truncated: false, info };
    }

    /*
    */
    render() {
        // clear canvas
        this.ctx.clearRect(0, 0, this.SCREEN_WIDTH, this.SCREEN_HEIGHT);

        // grass
        this.ctx.fillStyle = this.GRASS_COLOR;
        this.ctx.fillRect(0, 0, this.SCREEN_WIDTH, this.SCREEN_HEIGHT);

        // road
        this.ctx.fillStyle = this.DARK_ROAD_COLOR;
        this.ctx.fillRect(
            this.SCREEN_WIDTH / 2 - this.road_w / 2, 0,
            this.road_w, this.SCREEN_HEIGHT
        );

        // white lines
        this.ctx.fillStyle = this.WHITE_LINE_COLOR;
        this.ctx.fillRect(
            this.SCREEN_WIDTH / 2 - this.road_w / 2 + this.roadmark_w * 2, 0,
            this.roadmark_w, this.SCREEN_HEIGHT
        );
        this.ctx.fillRect(
            this.SCREEN_WIDTH / 2 + this.road_w / 2 - this.roadmark_w * 3, 0,
            this.roadmark_w, this.SCREEN_HEIGHT
        );

        // dashed yellow line
        this.ctx.fillStyle = this.YELLOW_LINE_COLOR;
        this.line_offset = (this.line_offset + this.speed) % (this.SCREEN_HEIGHT / 10);
        const dash_height = this.SCREEN_HEIGHT / 20;
        const dash_gap = this.SCREEN_HEIGHT / 10;
        
        for (let y = -dash_gap; y < this.SCREEN_HEIGHT; y += dash_gap) {
            this.ctx.fillRect(
                this.SCREEN_WIDTH / 2 - this.roadmark_w / 2, y + this.line_offset,
                this.roadmark_w, dash_height
            );
        }
        
        // cars
        this.ctx.drawImage(this.carImg, this.car_loc.x, this.car_loc.y, this.car_loc.width, this.car_loc.height);
        this.ctx.drawImage(this.car2Img, this.car2_loc.x, this.car2_loc.y, this.car2_loc.width, this.car2_loc.height);

    }
}