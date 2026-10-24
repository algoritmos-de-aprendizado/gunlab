package gunlab;

import java.awt.geom.Point2D;

import robocode.Rules;

/** Feature extraction; the column order matches gunlab.model.FEATURES in Python. */
public final class Features {
    public static final String[] NAMES = {
        "distance", "bullet_power", "lateral_speed", "advancing_velocity", "enemy_speed",
        "acceleration", "ticks_since_reverse", "wall_ahead", "wall_behind", "dist_last_10",
        "last_gf", "mean_gf_10",
        "hist_all_0", "hist_all_1", "hist_all_2", "hist_all_3", "hist_all_4",
        "hist_real_0", "hist_real_1", "hist_real_2", "hist_real_3", "hist_real_4"
    };
    public static final int COUNT = NAMES.length;
    public static final double WALL_MARGIN = 18.0;

    private static final int WALL_STEPS = 50;
    private static final int HISTORY = 10;
    private static final double MAX_TICKS = 200.0;

    private final Point2D.Double[] history = new Point2D.Double[HISTORY + 1];
    private int scans;
    private int lateralDirection = 1;
    private int ticksSinceReverse;
    private double previousSpeed;

    public int lateralDirection() {
        return lateralDirection;
    }

    public double[] update(Point2D.Double me, Point2D.Double enemy, double absBearing, double distance,
                           double enemyVelocity, double enemyHeading, double power,
                           double fieldWidth, double fieldHeight) {
        double lateral = enemyVelocity * Math.sin(enemyHeading - absBearing);
        double advancing = -enemyVelocity * Math.cos(enemyHeading - absBearing);
        int direction = lateral > 0 ? 1 : lateral < 0 ? -1 : lateralDirection;
        if (direction != lateralDirection) {
            lateralDirection = direction;
            ticksSinceReverse = 0;
        } else {
            ticksSinceReverse++;
        }
        double speed = Math.abs(enemyVelocity);
        double acceleration = scans == 0 ? 0.0 : speed - previousSpeed;
        previousSpeed = speed;

        history[scans % history.length] = enemy;
        Point2D.Double past = history[scans >= HISTORY ? (scans - HISTORY) % history.length : 0];
        scans++;

        double escape = maxEscapeAngle(power);
        double[] row = new double[COUNT];
        row[0] = distance;
        row[1] = power;
        row[2] = Math.abs(lateral);
        row[3] = advancing;
        row[4] = speed;
        row[5] = acceleration;
        row[6] = Math.min(ticksSinceReverse, MAX_TICKS);
        row[7] = wallFraction(me, absBearing, distance, direction, escape, fieldWidth, fieldHeight);
        row[8] = wallFraction(me, absBearing, distance, -direction, escape, fieldWidth, fieldHeight);
        row[9] = enemy.distance(past);
        History.fill(row, 10);
        return row;
    }

    public static double bulletPower(double distance, double enemyEnergy, double myEnergy) {
        double power = distance < 150.0 ? 3.0 : Math.min(3.0, Math.max(1.0, 450.0 / distance));
        power = Math.min(power, myEnergy / 8.0);
        power = Math.min(power, enemyEnergy / 4.0 + 0.1);
        return Math.max(Rules.MIN_BULLET_POWER, Math.min(Rules.MAX_BULLET_POWER, power));
    }

    public static double maxEscapeAngle(double power) {
        return Math.asin(Rules.MAX_VELOCITY / Rules.getBulletSpeed(power));
    }

    /** Fraction of the maximum escape angle reachable in one orbit direction before leaving the field. */
    public static double wallFraction(Point2D.Double source, double absBearing, double distance, int direction,
                                      double escape, double fieldWidth, double fieldHeight) {
        for (int i = 1; i <= WALL_STEPS; i++) {
            Point2D.Double p = project(source, absBearing + direction * escape * i / WALL_STEPS, distance);
            if (p.x < WALL_MARGIN || p.y < WALL_MARGIN
                    || p.x > fieldWidth - WALL_MARGIN || p.y > fieldHeight - WALL_MARGIN) {
                return (i - 1) / (double) WALL_STEPS;
            }
        }
        return 1.0;
    }

    public static Point2D.Double project(Point2D.Double p, double angle, double length) {
        return new Point2D.Double(p.x + Math.sin(angle) * length, p.y + Math.cos(angle) * length);
    }

    public static double absoluteBearing(Point2D.Double from, Point2D.Double to) {
        return Math.atan2(to.x - from.x, to.y - from.y);
    }
}
