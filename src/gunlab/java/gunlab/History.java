package gunlab;

/**
 * GFs of the waves that already broke in this battle. Static, so it survives the rounds of one battle;
 * Robocode gives each robot its own class loader per battle, so every battle starts empty.
 */
final class History {
    private static final int BINS = 5;
    private static final int RECENT = 10;
    private static final double ALL_DECAY = 0.998;
    private static final double REAL_DECAY = 0.95;

    private static final double[] recent = new double[RECENT];
    private static final double[] all = new double[BINS];
    private static final double[] real = new double[BINS];
    private static int count;

    private History() {
    }

    static void add(double gf, boolean isReal) {
        recent[count % RECENT] = gf;
        count++;
        accumulate(all, ALL_DECAY, gf);
        if (isReal) {
            accumulate(real, REAL_DECAY, gf);
        }
    }

    /** Writes last_gf, mean_gf_10, hist_all_0..4 and hist_real_0..4 into row, starting at offset. */
    static void fill(double[] row, int offset) {
        int n = Math.min(count, RECENT);
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += recent[i];
        }
        row[offset] = count == 0 ? 0.0 : recent[(count - 1) % RECENT];
        row[offset + 1] = n == 0 ? 0.0 : sum / n;
        normalized(all, row, offset + 2);
        normalized(real, row, offset + 2 + BINS);
    }

    private static void accumulate(double[] histogram, double decay, double gf) {
        for (int i = 0; i < BINS; i++) {
            histogram[i] *= decay;
        }
        histogram[Math.min(BINS - 1, (int) ((gf + 1.0) / 2.0 * BINS))] += 1.0;
    }

    private static void normalized(double[] histogram, double[] row, int offset) {
        double total = 0.0;
        for (double v : histogram) {
            total += v;
        }
        for (int i = 0; i < BINS; i++) {
            row[offset + i] = total > 0.0 ? histogram[i] / total : 1.0 / BINS;
        }
    }
}
