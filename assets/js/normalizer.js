// Add this class definition somewhere in your script
class ObservationNormalizer {
    constructor(stats) {
        // Check if stats are nested (e.g., from a converted .pkl)
        if (stats.obs_rms) {
            this.mean = new Float32Array(stats.obs_rms.mean);
            this.var = new Float32Array(stats.obs_rms.var);
        }
        // Check if stats are flat (e.g., from a custom export)
        else if (stats.mean) {
            this.mean = new Float32Array(stats.mean);
            this.var = new Float32Array(stats.var);
        } else {
            console.error("Stats file in unknown format. Looking for 'mean' or 'obs_rms.mean'.");
            return;
        }

        this.epsilon = stats.epsilon || 1e-8;
        this.clip_obs = stats.clip_obs || 10.0;
    }

    /**
     * Normalizes a single observation array.
     * @param {Array<number>} obs - The raw observation array.
     * @returns {Float32Array} The normalized observation array.
     */
    normalize(obs) {
        const normalizedObs = new Float32Array(obs.length);
        for (let i = 0; i < obs.length; i++) {
            // Apply normalization formula: (obs - mean) / sqrt(var + epsilon)
            let norm_val = (obs[i] - this.mean[i]) / Math.sqrt(this.var[i] + this.epsilon);

            // Apply clipping
            normalizedObs[i] = Math.max(-this.clip_obs, Math.min(this.clip_obs, norm_val));
        }
        return normalizedObs;
    }
}