package gunlab;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.geom.Point2D;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;

import robocode.BulletHitEvent;
import robocode.RobocodeFileOutputStream;

/**
 * Aims with the network in weights.txt (data directory). With painting enabled it draws, for each bullet,
 * its wave, the network's scores along the wave, and the predicted (red ring) versus actual (green dot) GF
 * when the wave reaches the enemy.
 */
public class NeuralBot extends GunBase {
    private static final String WEIGHTS = "weights.txt";
    private static final String STATS = "stats.txt";
    private static final int MARK_TICKS = 40;

    private static Mlp model;
    private static String loadError;
    private static int shots;
    private static int hits;
    private static int broken;
    private static int wouldHit;

    private final List<Mark> marks = new ArrayList<>();

    @Override
    protected void onRoundStart() {
        if (model != null || loadError != null) {
            return;
        }
        try {
            Mlp loaded = Mlp.load(getDataFile(WEIGHTS));
            if (loaded.inputs() != Features.COUNT) {
                throw new IOException("network has " + loaded.inputs() + " inputs, expected " + Features.COUNT);
            }
            model = loaded;
        } catch (IOException | RuntimeException ex) {
            loadError = String.valueOf(ex.getMessage());
            out.println("Weights not loaded: " + loadError);
            out.println("Expected at: " + getDataFile(WEIGHTS).getAbsolutePath());
        }
    }

    @Override
    protected double[] predict(double[] features) {
        return model == null ? new double[] {0.0} : model.forward(features);
    }

    @Override
    protected void onWaveCreated(Wave wave) {
        if (wave.real) {
            shots++;
        }
    }

    @Override
    protected void onWaveBreak(Wave wave, double gf, double width, long time) {
        broken++;
        if (Math.abs(wave.predictedGf - gf) <= width) {
            wouldHit++;
        }
        if (wave.real) {
            marks.add(new Mark(wave, gf, wave.radius(time), time + MARK_TICKS));
        }
    }

    @Override
    public void onBulletHit(BulletHitEvent event) {
        hits++;
    }

    /** Cumulative counters for this battle, read by gunlab after the battle. */
    @Override
    protected void onRoundFinished() {
        try {
            PrintStream stats = new PrintStream(new RobocodeFileOutputStream(getDataFile(STATS).getPath()));
            stats.printf(Locale.ROOT, "loaded %d%nshots %d%nhits %d%nwaves %d%nwould_hit %d%n",
                    model == null ? 0 : 1, shots, hits, broken, wouldHit);
            stats.close();
        } catch (IOException ex) {
            out.println("Cannot write " + STATS + ": " + ex.getMessage());
        }
    }

    @Override
    public void onPaint(Graphics2D g) {
        long time = getTime();
        g.setStroke(new BasicStroke(1.5f));
        for (Wave wave : waves) {
            if (!wave.real) {
                continue;
            }
            double r = wave.radius(time);
            g.setColor(new Color(255, 255, 255, 60));
            g.drawOval((int) (wave.source.x - r), (int) (wave.source.y - r), (int) (2 * r), (int) (2 * r));
            paintScores(g, wave, r);
        }
        Iterator<Mark> it = marks.iterator();
        while (it.hasNext()) {
            Mark mark = it.next();
            if (time > mark.expires) {
                it.remove();
                continue;
            }
            g.setColor(new Color(255, 255, 255, 120));
            g.drawLine((int) mark.predicted.x, (int) mark.predicted.y, (int) mark.actual.x, (int) mark.actual.y);
            g.setColor(Color.RED);
            g.drawOval((int) mark.predicted.x - 6, (int) mark.predicted.y - 6, 12, 12);
            g.setColor(Color.GREEN);
            g.fillOval((int) mark.actual.x - 4, (int) mark.actual.y - 4, 8, 8);
        }
        g.setFont(new Font(Font.MONOSPACED, Font.BOLD, 14));
        g.setColor(Color.WHITE);
        g.drawString(status(), 10, 10);
    }

    private String status() {
        if (model == null) {
            return getName() + "  no weights: " + loadError;
        }
        return String.format(Locale.ROOT, "%s  hits %d/%d (%.1f%%)  waves %d  would hit %.1f%%",
                getName(), hits, shots, 100.0 * hits / Math.max(1, shots),
                broken, 100.0 * wouldHit / Math.max(1, broken));
    }

    private static void paintScores(Graphics2D g, Wave wave, double r) {
        double[] y = wave.output;
        if (y.length == 1) {
            Point2D.Double p = Features.project(wave.source, wave.angleFor(wave.predictedGf), r);
            g.setColor(Color.ORANGE);
            g.fillOval((int) p.x - 4, (int) p.y - 4, 8, 8);
            return;
        }
        double max = Double.NEGATIVE_INFINITY;
        for (double v : y) {
            max = Math.max(max, v);
        }
        double[] p = new double[y.length];
        double sum = 0.0;
        for (int k = 0; k < y.length; k++) {
            p[k] = Math.exp(y[k] - max);
            sum += p[k];
        }
        double top = 0.0;
        for (int k = 0; k < y.length; k++) {
            p[k] /= sum;
            top = Math.max(top, p[k]);
        }
        for (int k = 0; k < y.length; k++) {
            double q = p[k] / top;
            Point2D.Double point = Features.project(wave.source, wave.angleFor(-1.0 + 2.0 * k / (y.length - 1)), r);
            int size = 2 + (int) Math.round(6 * q);
            g.setColor(new Color(255, (int) (220 * (1 - q)), 0, (int) (40 + 215 * q)));
            g.fillOval((int) point.x - size / 2, (int) point.y - size / 2, size, size);
        }
    }

    private static final class Mark {
        final Point2D.Double predicted;
        final Point2D.Double actual;
        final long expires;

        Mark(Wave wave, double gf, double radius, long expires) {
            this.predicted = Features.project(wave.source, wave.angleFor(wave.predictedGf), radius);
            this.actual = Features.project(wave.source, wave.angleFor(gf), radius);
            this.expires = expires;
        }
    }
}
