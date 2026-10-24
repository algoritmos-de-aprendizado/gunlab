package gunlab;

import java.awt.geom.Point2D;

public final class Wave {
    public final Point2D.Double source;
    public final long fireTime;
    public final double speed;
    public final double absBearing;
    public final int direction;
    public final double maxEscapeAngle;
    public final double[] features;
    public final boolean real;
    public final double predictedGf;
    public final double[] output;

    Wave(Point2D.Double source, long fireTime, double speed, double absBearing, int direction,
         double maxEscapeAngle, double[] features, boolean real, double predictedGf, double[] output) {
        this.source = source;
        this.fireTime = fireTime;
        this.speed = speed;
        this.absBearing = absBearing;
        this.direction = direction;
        this.maxEscapeAngle = maxEscapeAngle;
        this.features = features;
        this.real = real;
        this.predictedGf = predictedGf;
        this.output = output;
    }

    public double radius(long time) {
        return (time - fireTime) * speed;
    }

    public double angleFor(double gf) {
        return absBearing + direction * gf * maxEscapeAngle;
    }
}
