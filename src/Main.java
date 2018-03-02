import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import java.util.stream.Stream;

/*
 * Основен клас на приложението.
 */
public class Main {

	/*
	 * Стартова точка на програмата.
	 */
	public static void main(String[] args) {
		/*
		 * Данните от времевия ред се зареждат от служебен файл. Всяка стойност е
		 * разделена със запетая.
		 */
		double dst[] = Stream.of(Messages.getString("Main.0").split(",")).mapToDouble(Double::parseDouble).toArray();

		Map<Experiment.Parameters, Double> statistics = new HashMap<Experiment.Parameters, Double>();

		Scanner in = new Scanner(System.in);
		do {
			Experiment.Parameters parameters = null;
			do {
				parameters = new Experiment.Parameters().randomize(dst);
			} while (statistics.containsKey(parameters));

			try {
				Experiment experiment = new Experiment(parameters, dst);
				for (int e = 0; e < Constants.NUMBER_OF_EXPERIMENTS; e++) {
					experiment.initialize();
					experiment.train();
					statistics.put(parameters, experiment.test());
				}
				statistics.put(parameters, statistics.get(parameters) / Constants.NUMBER_OF_EXPERIMENTS);
				System.out.println("Parameters: " + parameters);
				System.out.println("Average Error: " + statistics.get(parameters));
			} catch (Exception exception) {
				/*
				 * Нещо не е наред при някои конфигурации за обучение.
				 */
			}
		} while (true);
	}
}
