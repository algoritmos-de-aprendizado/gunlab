package gunlab;

import java.awt.Color;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Locale;

import robocode.RobocodeFileOutputStream;

/**
 * Fires real bullets at random GFs (so wave-surfing opponents react as in the arena)
 * and appends one row per broken wave to rows.csv in its data directory.
 */
public class LoggerBot extends GunBase {
    private PrintStream csv;
    private double aimGf;

    @Override
    protected void onRoundStart() {
        aimGf = 2.0 * Math.random() - 1.0;
        try {
            String path = getDataFile("rows.csv").getPath();
            csv = new PrintStream(new BufferedOutputStream(new RobocodeFileOutputStream(path, true)));
        } catch (IOException ex) {
            out.println("Cannot open rows.csv: " + ex.getMessage());
        }
    }

    @Override
    protected double[] predict(double[] features) {
        return new double[] {aimGf};
    }

    @Override
    protected void onWaveCreated(Wave wave) {
        if (wave.real) {
            aimGf = 2.0 * Math.random() - 1.0;
        }
    }

    @Override
    protected void onWaveBreak(Wave wave, double gf, double width, long time) {
        if (csv == null) {
            return;
        }
        StringBuilder row = new StringBuilder();
        for (double value : wave.features) {
            row.append(String.format(Locale.ROOT, "%.5f,", value));
        }
        row.append(String.format(Locale.ROOT, "%.5f,%.5f,%d,%d", gf, width, wave.real ? 1 : 0, getRoundNum()));
        csv.println(row);
    }

    @Override
    protected void onRoundFinished() {
        if (csv != null) {
            csv.close();
            csv = null;
        }
    }

    @Override
    protected Color bodyColor() {
        return Color.GRAY;
    }
}
