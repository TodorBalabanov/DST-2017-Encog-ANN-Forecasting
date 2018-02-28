import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.neural.data.NeuralDataSet;
import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.Train;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.util.arrayutil.NormalizeArray;

public class Experiment {

	public static class Parameters {
		/*
		 * ������ �� ��������� � ��������, ����� �� ������ �� ����� ��
		 * ������������ �������� �����.
		 */
		private final static int DEFAULT_LAG_LENGTH = 1;

		/*
		 * ������ �� ��������� � ��������, ����� �� ������ ���� �������� ��
		 * ������������ �������� �����.
		 */
		private final static int DEFAULT_LEAD_LENGTH = 1;

		/*
		 * ��� ��� � �������� � ������������ ��������� ����� �� ��� ������ ��
		 * ���� ����� ����, �� ��������� �� �������� ������ �� ������ � ����� �
		 * ����� ��������� ������ ������ �� ���������� ����.
		 */
		private final static int DEFAULT_HIDDEN_LENGTH[] = { 1 };

		/*
		 * ���� �� ��������� �� ��������� �� �������� �� ������������ ��������
		 * �����.
		 */
		private final static double DEFAULT_TRAINING_PART = 0.5;

		/*
		 * ���������, ����� �� �� ��������� �� ��������, �� ������� �� ���������
		 * �� ���������� ���������� �� ������������ ��������� �����.
		 */
		private final static double DEFAULT_TESTING_PART = 1.0 - DEFAULT_TRAINING_PART;

		private int lagLength = DEFAULT_LAG_LENGTH;

		private int leadLength = DEFAULT_LEAD_LENGTH;

		private int hiddenLength[] = DEFAULT_HIDDEN_LENGTH;

		private double trainingPart = DEFAULT_TRAINING_PART;

		private double testingPart = DEFAULT_TESTING_PART;

		public Parameters randomize(double[] series) {
			/*
			 * ������ ������� �� ����� ��������� �� ���� �� ��������� ���������
			 * �� �������� ���.
			 */
			do {
				lagLength = 1 + Constants.PRNG.nextInt(series.length);
				leadLength = 1 + Constants.PRNG.nextInt(series.length);
			} while ((lagLength + leadLength) >= series.length);

			trainingPart = Constants.PRNG.nextDouble();
			testingPart = 1D - trainingPart;

			hiddenLength = new int[Constants.MIN_HIDDEN_NUMBER
					+ Constants.PRNG.nextInt(Constants.MAX_HIDDEN_NUBMER - Constants.MIN_HIDDEN_NUMBER + 1)];

			for (int i = 0; i < hiddenLength.length; i++) {
				hiddenLength[i] = Constants.MIN_HIDDEN_LENGTH
						+ Constants.PRNG.nextInt(Constants.MAX_HIDDEN_LENGTH - Constants.MIN_HIDDEN_LENGTH + 1);
			}

			return this;
		}

		public int lagLength() {
			return lagLength;
		}

		public void lagLength(int lagLength) {
			this.lagLength = lagLength;
		}

		public int leadLength() {
			return leadLength;
		}

		public void leadLength(int leadLength) {
			this.leadLength = leadLength;
		}

		public int[] hiddenLength() {
			return hiddenLength;
		}

		public void hiddenLength(int[] hiddenLength) {
			this.hiddenLength = hiddenLength;
		}

		public double trainingPart() {
			return trainingPart;
		}

		public void trainingPart(double trainingPart) {
			this.trainingPart = trainingPart;
		}

		public double testingPart() {
			return testingPart;
		}

		public void testingPart(double testingPart) {
			this.testingPart = testingPart;
		}

		@Override
		public int hashCode() {
			final int prime = 31;
			int result = 1;
			result = prime * result + Arrays.hashCode(hiddenLength);
			result = prime * result + lagLength;
			result = prime * result + leadLength;
			long temp;
			temp = Double.doubleToLongBits(testingPart);
			result = prime * result + (int) (temp ^ (temp >>> 32));
			temp = Double.doubleToLongBits(trainingPart);
			result = prime * result + (int) (temp ^ (temp >>> 32));
			return result;
		}

		@Override
		public boolean equals(Object obj) {
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			if (getClass() != obj.getClass())
				return false;
			Parameters other = (Parameters) obj;
			if (!Arrays.equals(hiddenLength, other.hiddenLength))
				return false;
			if (lagLength != other.lagLength)
				return false;
			if (leadLength != other.leadLength)
				return false;
			if (Double.doubleToLongBits(testingPart) != Double.doubleToLongBits(other.testingPart))
				return false;
			if (Double.doubleToLongBits(trainingPart) != Double.doubleToLongBits(other.trainingPart))
				return false;
			return true;
		}

		@Override
		public String toString() {
			return "[lag=" + lagLength + ", lead=" + leadLength + ", hidden=" + Arrays.toString(hiddenLength)
					+ ", training=" + (int) (100 * trainingPart + 0.5) + "%, testing=" + (int) (100 * testingPart + 0.5)
					+ "%]";
		}
	}

	private Parameters parameters = null;

	private NeuralDataSet trainingSet = new BasicNeuralDataSet();

	private NeuralDataSet testingSet = new BasicNeuralDataSet();

	private BasicNetwork network = null;

	public Experiment(Parameters parameters, double[] series) {
		this.parameters = parameters;

		/*
		 * ��� ������������ ��������� ������������ ������� � �������� �� ��
		 * ��������� ������� � ������ ����, ���� �� �������� ����� �� �����
		 * ��������� ����������, ������ ���� ����.
		 */
		double range[] = { -Double.MAX_VALUE, +Double.MAX_VALUE };
		Constants.ACTIVATION.activationFunction(range, 0, range.length);

		/*
		 * ������� � �� �� �� ��������� �������� ���������, � �� �� ����
		 * �������� ������. ���� ��������� ������� �� �������� � ���� ���� ����
		 * ���������� �� �� ����������� ��-������ � ��-����� ��������� ���
		 * �������� ���, ����� �� ���� �� �� ���� �����������, �� �� � ���������
		 * �� �� ������ � ������.
		 */
		range[0] += 0.1;
		range[1] -= 0.1;

		/*
		 * ��������� ��� �� ������������� � ��������� ����� � �������� ��
		 * ������������ ������������ �������. ���� ��������� ������������������
		 * �� ������������ �������� �����. �� �� �� �������� ������������ ��
		 * ������ �� ������������ ����� ������ �� �� ������� ��������� ��������
		 * �� ���������������.
		 */
		NormalizeArray normalizer = new NormalizeArray();
		normalizer.setNormalizedLow(range[0]);
		normalizer.setNormalizedHigh(range[1]);
		double scaled[] = normalizer.process(series);

		/*
		 * ��������� ��� �� ������� �� ����� ������� ����� �������� ������� ��
		 * ���� � ��������� �� ������ ���� �� ��������� �� ������� � �������
		 * ��������. ���� ����������� ������� �� ����� ��������� � ��� ����� ��
		 * �������� � �� �����������.
		 */
		List<Object> samples = new ArrayList<Object>();
		for (int a = 0, b = parameters.lagLength; a < series.length - parameters.lagLength - parameters.leadLength
				&& b < series.length - parameters.lagLength; a++, b++) {
			double lag[] = Arrays.copyOfRange(series, a, a + parameters.lagLength);
			double lead[] = Arrays.copyOfRange(series, b, b + parameters.leadLength);
			samples.add(new double[][] { lag, lead });
		}

		/*
		 * ��������� �� ���������� �� ������� �������, ���� �� � ����� ���������
		 * (�� �������� � �� �����������) �� �������� �������� �������.
		 * ����������� �� ����� ������������ ��������� �� ���� ������� �� ������
		 * ������ ���������� �����������.
		 */
		Collections.shuffle(samples);

		/*
		 * ��������� �� �������� �� ������������ � ������������.
		 */
		for (int i = 0; i < samples.size(); i++) {
			double sample[][] = (double[][]) samples.get(i);

			MLData input = new BasicMLData(sample[0]);
			MLData ideal = new BasicMLData(sample[1]);
			MLDataPair pair = new BasicMLDataPair(input, ideal);

			if ((double) i / samples.size() < parameters.trainingPart) {
				trainingSet.add(pair);
			} else {
				testingSet.add(pair);
			}
		}
	}

	public void initialize() {
		/*
		 * ������������� ���������� �� ������ �� ��� ��� ������ ������. ��������
		 * ���� ���������� ������ ��������� �� �������� �����. �������� ��
		 * ������ �� ���� ������ ������.
		 */
		network = new BasicNetwork();
		network.addLayer(new BasicLayer(Constants.ACTIVATION, true, parameters.lagLength));
		for (int size : parameters.hiddenLength) {
			network.addLayer(new BasicLayer(Constants.ACTIVATION, true, size));
		}
		network.addLayer(new BasicLayer(Constants.ACTIVATION, false, parameters.leadLength));
		network.getStructure().finalizeStructure();
		network.reset();

		/*
		 * ������ ��������� �� ����� Java ��������� �������������� ���������� ��
		 * ���������� ��� �� ����� ���� ������ ����������, ���� �� ���������� ��
		 * ������� �� �������� �� ���� ��-�����.
		 */
		try {
			(new ResilientPropagation(network, new BasicNeuralDataSet())).iteration();
		} catch (Exception exception) {
		}
	}

	public void train() {

		/*
		 * ��������� �� ���� ����� ��������, ���� �� ������������ �������� �����
		 * �� ���� ������ ��������������.
		 */
		Train train = new ResilientPropagation(network, trainingSet);
		train.iteration();

		/*
		 * �������� �� ������������ ���������� �� ��������� �������� �� �����.
		 */
		int epoch = 0;
		for (long stop = System.currentTimeMillis() + Constants.MAX_TRAINING_MILLISECONDS; System
				.currentTimeMillis() < stop;) {
			long start = System.currentTimeMillis();

			do {
				train.iteration();
				epoch++;
			} while ((System.currentTimeMillis() - start) < Constants.SINGLE_MEASUREMENT_MILLISECONDS);

			System.err.print(System.currentTimeMillis() - start);
			System.err.print("\t");
			System.err.print(epoch);
			System.err.print("\t");
			System.err.print(train.getError());
			System.err.print("\n");
		}
		System.err.println();
	}

	public double test() {
		/*
		 * ���������� �� �������� ����� ������������ �������� ����� �������
		 * ����� ������������ ���������.
		 */
		System.err.println("Training error:\t" + network.calculateError(trainingSet));
		System.err.println("Testing error:\t" + network.calculateError(testingSet));
		for (MLDataPair pair : testingSet) {
			// MLData output = network.compute(pair.getInput());
			// System.err.print(pair.getIdeal());
			// System.err.print("\t");
			// System.err.print(output);
			// System.err.print("\n");
		}

		return network.calculateError(testingSet);
	}

	public void forecast(double[] series) {
		/*
		 * ������ ��������, ����� � ������������� ��� ������������ �����.
		 */
		double[] forecast = new double[parameters.leadLength + 2];
		NormalizeArray normalizer = new NormalizeArray();
		network.compute(
				normalizer.process(Arrays.copyOfRange(series, series.length - parameters.lagLength, series.length)),
				forecast);
		forecast[forecast.length - 2] = -Double.MAX_VALUE;
		forecast[forecast.length - 1] = +Double.MAX_VALUE;
		Constants.ACTIVATION.activationFunction(forecast, forecast.length - 2, 2);
		normalizer.setNormalizedLow(Arrays.stream(series).min().getAsDouble());
		normalizer.setNormalizedHigh(Arrays.stream(series).max().getAsDouble());
		forecast = Arrays.copyOfRange(normalizer.process(forecast), 0, forecast.length - 2);
		System.err.println("Forecast:\t" + Arrays.toString(forecast));
	}

}
