package gunlab;

import java.awt.Color;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import robocode.AdvancedRobot;
import robocode.BattleEndedEvent;
import robocode.DeathEvent;
import robocode.HitWallEvent;
import robocode.RoundEndedEvent;
import robocode.Rules;
import robocode.ScannedRobotEvent;
import robocode.WinEvent;
import robocode.util.Utils;

/**
 * Fixed movement (orbit with random reversals and wall smoothing), radar lock, features and waves.
 * Subclasses only decide where to aim, through predict().
 */
public abstract class GunBase extends AdvancedRobot {
    private static final double PREFERRED_DISTANCE = 450.0;
    private static final double WALL_STICK = 160.0;

    protected final List<Wave> waves = new ArrayList<>();
    private final Features tracker = new Features();
    private Rectangle2D.Double field;
    private int moveDirection = 1;
    private double enemyEnergy = 100.0;

    /** Model output: one value (the GF itself) or K > 1 scores, one per GF bin. */
    protected abstract double[] predict(double[] features);

    protected void onRoundStart() {
    }

    protected void onWaveCreated(Wave wave) {
    }

    /** width: the enemy's half-width in GF units when the wave broke. */
    protected void onWaveBreak(Wave wave, double gf, double width, long time) {
    }

    protected void onRoundFinished() {
    }

    protected Color bodyColor() {
        return Color.getHSBColor((getClass().getName().hashCode() & 0xffff) / 65536f, 0.75f, 0.95f);
    }

    @Override
    public void run() {
        field = new Rectangle2D.Double(Features.WALL_MARGIN, Features.WALL_MARGIN,
                getBattleFieldWidth() - 2 * Features.WALL_MARGIN, getBattleFieldHeight() - 2 * Features.WALL_MARGIN);
        Color color = bodyColor();
        setColors(color, color.darker(), Color.WHITE);
        setBulletColor(color);
        setAdjustGunForRobotTurn(true);
        setAdjustRadarForGunTurn(true);
        onRoundStart();
        while (true) {
            if (getRadarTurnRemainingRadians() == 0.0) {
                setTurnRadarRightRadians(Double.POSITIVE_INFINITY);
            }
            execute();
        }
    }

    @Override
    public void onScannedRobot(ScannedRobotEvent e) {
        long time = getTime();
        Point2D.Double me = new Point2D.Double(getX(), getY());
        double absBearing = getHeadingRadians() + e.getBearingRadians();
        double distance = e.getDistance();
        Point2D.Double enemy = Features.project(me, absBearing, distance);

        setTurnRadarRightRadians(Utils.normalRelativeAngle(absBearing - getRadarHeadingRadians()) * 2.0);
        breakWaves(enemy, time);

        double power = Features.bulletPower(distance, e.getEnergy(), getEnergy());
        double[] features = tracker.update(me, enemy, absBearing, distance, e.getVelocity(), e.getHeadingRadians(),
                power, getBattleFieldWidth(), getBattleFieldHeight());

        boolean fired = false;
        if (getGunHeat() == 0.0 && Math.abs(getGunTurnRemainingRadians()) < Math.atan2(18.0, distance)) {
            fired = setFireBullet(power) != null;
        }

        double[] output = predict(features);
        if (output == null || output.length == 0) {
            output = new double[] {0.0};
        }
        double gf = Mlp.guessFactor(output);
        Wave wave = new Wave(me, time, Rules.getBulletSpeed(power), absBearing, tracker.lateralDirection(),
                Features.maxEscapeAngle(power), features, fired, gf, output);
        waves.add(wave);
        onWaveCreated(wave);
        setTurnGunRightRadians(Utils.normalRelativeAngle(wave.angleFor(gf) - getGunHeadingRadians()));

        move(me, absBearing, distance, e.getEnergy());
    }

    private void breakWaves(Point2D.Double enemy, long time) {
        Iterator<Wave> it = waves.iterator();
        while (it.hasNext()) {
            Wave w = it.next();
            if (w.radius(time) >= w.source.distance(enemy)) {
                double offset = Utils.normalRelativeAngle(Features.absoluteBearing(w.source, enemy) - w.absBearing);
                double gf = Math.max(-1.0, Math.min(1.0, offset * w.direction / w.maxEscapeAngle));
                double width = Math.atan(18.0 / w.source.distance(enemy)) / w.maxEscapeAngle;
                History.add(gf, w.real);
                onWaveBreak(w, gf, width, time);
                it.remove();
            }
        }
    }

    private void move(Point2D.Double me, double absBearing, double distance, double energy) {
        double drop = enemyEnergy - energy;
        enemyEnergy = energy;
        boolean enemyFired = drop >= Rules.MIN_BULLET_POWER && drop <= Rules.MAX_BULLET_POWER;
        if ((enemyFired && Math.random() < 0.5) || Math.random() < 0.01) {
            moveDirection = -moveDirection;
        }
        double approach = Math.max(-0.5, Math.min(0.5, (distance - PREFERRED_DISTANCE) / 400.0));
        double angle = absBearing + moveDirection * (Math.PI / 2.0 - approach);
        for (int i = 0; i < 100 && !field.contains(Features.project(me, angle, WALL_STICK)); i++) {
            angle += moveDirection * 0.05;
        }
        double turn = Utils.normalRelativeAngle(angle - getHeadingRadians());
        double ahead = 100.0;
        if (Math.abs(turn) > Math.PI / 2.0) {
            turn = Utils.normalRelativeAngle(turn + Math.PI);
            ahead = -ahead;
        }
        setTurnRightRadians(turn);
        setAhead(ahead);
    }

    @Override
    public void onHitWall(HitWallEvent e) {
        moveDirection = -moveDirection;
    }

    @Override
    public void onDeath(DeathEvent e) {
        onRoundFinished();
    }

    @Override
    public void onWin(WinEvent e) {
        onRoundFinished();
    }

    @Override
    public void onRoundEnded(RoundEndedEvent e) {
        onRoundFinished();
    }

    @Override
    public void onBattleEnded(BattleEndedEvent e) {
        onRoundFinished();
    }
}
