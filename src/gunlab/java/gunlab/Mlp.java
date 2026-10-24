package gunlab;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** Forward pass for the weights format written by gunlab.model.save_weights in Python. */
public final class Mlp {
    private static final List<String> ACTIVATIONS = Arrays.asList("identity", "tanh", "logistic", "relu");

    private final double[] mean;
    private final double[] std;
    private final double[][][] weights;
    private final double[][] biases;
    private final String[] activations;

    private Mlp(double[] mean, double[] std, double[][][] weights, double[][] biases, String[] activations) {
        this.mean = mean;
        this.std = std;
        this.weights = weights;
        this.biases = biases;
        this.activations = activations;
    }

    public int inputs() {
        return mean.length;
    }

    public double[] forward(double[] x) {
        double[] a = new double[mean.length];
        for (int i = 0; i < a.length; i++) {
            a[i] = (x[i] - mean[i]) / std[i];
        }
        for (int layer = 0; layer < weights.length; layer++) {
            double[][] w = weights[layer];
            double[] next = new double[w.length];
            for (int o = 0; o < w.length; o++) {
                double sum = biases[layer][o];
                for (int i = 0; i < a.length; i++) {
                    sum += w[o][i] * a[i];
                }
                next[o] = activate(activations[layer], sum);
            }
            a = next;
        }
        return a;
    }

    /** K = 1: the output is the GF. K > 1: center of the highest-scoring of K equal bins. */
    public static double guessFactor(double[] output) {
        if (output.length == 1) {
            return Math.max(-1.0, Math.min(1.0, output[0]));
        }
        int best = 0;
        for (int k = 1; k < output.length; k++) {
            if (output[k] > output[best]) {
                best = k;
            }
        }
        return -1.0 + 2.0 * best / (output.length - 1);
    }

    private static double activate(String name, double z) {
        if ("tanh".equals(name)) {
            return Math.tanh(z);
        }
        if ("logistic".equals(name)) {
            return 1.0 / (1.0 + Math.exp(-z));
        }
        if ("relu".equals(name)) {
            return Math.max(0.0, z);
        }
        return z;
    }

    public static Mlp load(File file) throws IOException {
        Tokens t = new Tokens(file);
        t.expect("gunlab-mlp");
        t.expect("1");
        t.expect("inputs");
        int inputs = t.nextInt();
        t.expect("mean");
        double[] mean = t.nextDoubles(inputs);
        t.expect("std");
        double[] std = t.nextDoubles(inputs);
        t.expect("layers");
        int layers = t.nextInt();
        if (layers < 1) {
            throw new IOException("the network needs at least one layer");
        }
        double[][][] weights = new double[layers][][];
        double[][] biases = new double[layers][];
        String[] activations = new String[layers];
        int width = inputs;
        for (int layer = 0; layer < layers; layer++) {
            t.expect("dense");
            int in = t.nextInt();
            int out = t.nextInt();
            if (in != width) {
                throw new IOException("layer " + (layer + 1) + " has " + in + " inputs, expected " + width);
            }
            activations[layer] = t.next();
            if (!ACTIVATIONS.contains(activations[layer])) {
                throw new IOException("unknown activation " + activations[layer]);
            }
            weights[layer] = new double[out][];
            for (int o = 0; o < out; o++) {
                weights[layer][o] = t.nextDoubles(in);
            }
            biases[layer] = t.nextDoubles(out);
            width = out;
        }
        if (t.hasNext()) {
            throw new IOException("unexpected data after the last layer");
        }
        return new Mlp(mean, std, weights, biases, activations);
    }

    private static final class Tokens {
        private final List<String> items = new ArrayList<>();
        private int position;

        Tokens(File file) throws IOException {
            BufferedReader reader = new BufferedReader(new FileReader(file));
            try {
                String line;
                while ((line = reader.readLine()) != null) {
                    int comment = line.indexOf('#');
                    if (comment >= 0) {
                        line = line.substring(0, comment);
                    }
                    for (String token : line.trim().split("\\s+")) {
                        if (!token.isEmpty()) {
                            items.add(token);
                        }
                    }
                }
            } finally {
                reader.close();
            }
        }

        boolean hasNext() {
            return position < items.size();
        }

        String next() throws IOException {
            if (!hasNext()) {
                throw new IOException("unexpected end of file");
            }
            return items.get(position++);
        }

        void expect(String word) throws IOException {
            String token = next();
            if (!word.equals(token)) {
                throw new IOException("expected '" + word + "', found '" + token + "'");
            }
        }

        int nextInt() throws IOException {
            try {
                return Integer.parseInt(next());
            } catch (NumberFormatException ex) {
                throw new IOException("expected an integer", ex);
            }
        }

        double[] nextDoubles(int n) throws IOException {
            double[] values = new double[n];
            for (int i = 0; i < n; i++) {
                try {
                    values[i] = Double.parseDouble(next());
                } catch (NumberFormatException ex) {
                    throw new IOException("expected a number", ex);
                }
            }
            return values;
        }
    }
}
