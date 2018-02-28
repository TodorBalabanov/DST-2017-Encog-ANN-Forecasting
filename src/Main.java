import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import org.encog.engine.network.activation.ActivationFunction;
import org.encog.engine.network.activation.ActivationTANH;
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

/*
 * Основен клас на приложението.
 */
public class Main {

	/*
	 * Стартова точка на програмата.
	 */
	public static void main(String[] args) {
		/*
		 * Данните от времевия ред се зареждат от служебен файл. Всяка стойност
		 * е разделена със запетая.
		 */
		double dst[] = Stream.of(Messages.getString("Main.0").split(",")).mapToDouble(Double::parseDouble).toArray();

		Map<Experiment.Parameters, Double> statistics = new HashMap<Experiment.Parameters, Double>();

		Experiment.Parameters parameters = null;
		do {
			parameters = new Experiment.Parameters().randomize(dst);
		} while (statistics.containsKey(parameters));

		System.out.println("Parameters: " + parameters);
		Experiment experiment = new Experiment(parameters, dst);
		for (int e = 0; e < Constants.NUMBER_OF_EXPERIMENTS; e++) {
			experiment.initialize();
			experiment.train();
			statistics.put(parameters, experiment.test());
		}
		statistics.put(parameters, statistics.get(parameters)/Constants.NUMBER_OF_EXPERIMENTS);
		System.out.println("Average Error: " + statistics.get(parameters));
	}
}
