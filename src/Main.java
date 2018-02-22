import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
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
	 * Горна граница, в милисекунди, за време на обучение.
	 */
	private static final long MAX_TRAINING_MILLISECONDS = 10000;

	/*
	 * Интервал, в милисекунди, през който ще се очита прогреса на обучението.
	 */
	private static final long SINGLE_MEASUREMENT_MILLISECONDS = 100;

	/*
	 * Размер на прозореца в миналото, който се подава на входа на изкуствената
	 * невронна мрежа.
	 */
	private final static int LAG_LENGTH = 75;

	/*
	 * Размер на прозореца в бъдещето, който се очаква като прогноза от изкуствената
	 * невронна мрежа.
	 */
	private final static int LEAD_LENGTH = 5;

	/*
	 * Тъй кто е възможно в изкуствената неверонна мрежа да има повече от един скрит
	 * слой, то размерите на скритите слоеве се задава с масив и всяка стойностп
	 * оказва размер на съответния слой.
	 */
	private final static int HIDDEN_LENGTH[] = { 10 };

	/*
	 * Част от примерите се използват за обучение на изкуствената невронна мрежа.
	 */
	private final static double TRAINING_PART = 0.5;

	/*
	 * Примерите, които не се използват за обучение, се ползват за валидация на
	 * прогнозите генерирани от изкуствената неверонна мрежа.
	 */
	private final static double TESTING_PART = 1.0 - TRAINING_PART;

	/*
	 * Избор на активационна функция. Направено е така за да може с лекота да се
	 * сменят различните активационни функции и да се наблюдават евентуални разлики
	 * при процеса на обучение, идващи от особеносттите на активационната функция.
	 */
	private final static ActivationFunction ACTIVATION = new ActivationTANH();

	/*
	 * Стартова точка на програмата.
	 */
	public static void main(String[] args) {
		/*
		 * Данните от времевия ред се зареждат от служебен файл. Всяка стойност е
		 * разделена със запетая.
		 */
		double dst[] = Stream.of(Messages.getString("Main.0").split(",")).mapToDouble(Double::parseDouble).toArray();

		/*
		 * При асимптотично сходимите активационни функции е възможно да се определят
		 * горното и долнот ниво, така че входните данни да бъдат подходящо мащабирани,
		 * спрямо тези нива.
		 */
		double range[] = { -Double.MAX_VALUE, +Double.MAX_VALUE };
		ACTIVATION.activationFunction(range, 0, range.length);

		/*
		 * Разумно е да не се използват крайните стойности, а да се даде известен
		 * отстъп. Това подобрява процеса на обучение и също така дава възможност да се
		 * прогнозират по-високи и по-ниски стойности във времевия ред, които до сега не
		 * са били наблюдавани, но не е изключено да се появят в бъдеще.
		 */
		range[0] += 0.1;
		range[1] -= 0.1;

		/*
		 * Времевият ред се преоразмерява в диапазона който е подходящ за използваната
		 * активационна функция. Това подобрява производителността на изкуствената
		 * невронна мрежа. За да се използва информацията от изхода на изкуствената
		 * мрежа трябва да се приложи обратната операция за преоразмеряване.
		 */
		NormalizeArray normalizer = new NormalizeArray();
		normalizer.setNormalizedLow(range[0]);
		normalizer.setNormalizedHigh(range[1]);
		double scaled[] = normalizer.process(dst);

		/*
		 * Времевият ред се разделя на групи примери които съдържат парчета от реда с
		 * размерите на общата сума от дължината на миналия и бъдещия прозорец. Така
		 * формираните примери ще бъдат разделени в две групи за обучение и за
		 * верификация.
		 */
		List<Object> samples = new ArrayList<Object>();
		for (int a = 0, b = LAG_LENGTH; a < scaled.length - LAG_LENGTH - LEAD_LENGTH
				&& b < scaled.length - LEAD_LENGTH; a++, b++) {
			double lag[] = Arrays.copyOfRange(scaled, a, a + LAG_LENGTH);
			double lead[] = Arrays.copyOfRange(scaled, b, b + LEAD_LENGTH);
			samples.add(new double[][] { lag, lead });
		}

		/*
		 * Примерите се разбъркват на случаен принцип, така че в двете множества (за
		 * обучение и за верификация) да попаднат различни примери. Постигането на добро
		 * разнообразие предпазва от това мрежата да изгуби своите обобщаващи
		 * възможности.
		 */
		Collections.shuffle(samples);

		/*
		 * Примерите се разделят на тренировъчни и проверовачни.
		 */
		NeuralDataSet trainingSet = new BasicNeuralDataSet();
		NeuralDataSet testingSet = new BasicNeuralDataSet();
		for (int i = 0; i < samples.size(); i++) {
			double sample[][] = (double[][]) samples.get(i);

			MLData input = new BasicMLData(sample[0]);
			MLData ideal = new BasicMLData(sample[1]);
			MLDataPair pair = new BasicMLDataPair(input, ideal);

			if ((double) i / samples.size() < TRAINING_PART) {
				trainingSet.add(pair);
			} else {
				testingSet.add(pair);
			}
		}

		/*
		 * Многослойният перцептрон се състои от три или повече слоеве. Входният слой
		 * единствено приема сигналите от външната среда. Възможни са повече от един
		 * скрити слоеве.
		 */
		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(ACTIVATION, true, LAG_LENGTH));
		for (int size : HIDDEN_LENGTH) {
			network.addLayer(new BasicLayer(ACTIVATION, true, size));
		}
		network.addLayer(new BasicLayer(ACTIVATION, false, LEAD_LENGTH));
		network.getStructure().finalizeStructure();
		network.reset();

		/*
		 * Поради механизма по който Java обработва първоначалното стартиране на
		 * програмния код се прави едно празно превъртане, така че засичането на времето
		 * за обучение да бъде по-точно.
		 */
		try {
			(new ResilientPropagation(network, new BasicNeuralDataSet())).iteration();
		} catch (Exception exception) {
		}

		/*
		 * Завъртане на един цикъл обучение, така че изкуствената невронна мрежа да бъде
		 * изцяло инициализирана.
		 */
		Train train = new ResilientPropagation(network, trainingSet);
		train.iteration();

		/*
		 * Обучение на многослойния перцептрон за определен интервал от време.
		 */
		int epoch = 0;
		for (long stop = System.currentTimeMillis() + MAX_TRAINING_MILLISECONDS; System.currentTimeMillis() < stop;) {
			long start = System.currentTimeMillis();

			do {
				train.iteration();
				epoch++;
			} while ((System.currentTimeMillis() - start) < SINGLE_MEASUREMENT_MILLISECONDS);

			System.out.print(System.currentTimeMillis() - start);
			System.out.print("\t");
			System.out.print(epoch);
			System.out.print("\t");
			System.out.print(train.getError());
			System.out.print("\n");
		}
		System.out.println();

		/*
		 * Определяне на грешката която изкуствената невронна мрежа допуска върху
		 * валидиращото множество.
		 */
		System.out.println("Training error:\t" + network.calculateError(trainingSet));
		System.out.println("Testing error:\t" + network.calculateError(testingSet));
		for (MLDataPair pair : testingSet) {
			// MLData output = network.compute(pair.getInput());
			// System.out.print(pair.getIdeal());
			// System.out.print("\t");
			// System.out.print(output);
			// System.out.print("\n");
		}

		/*
		 * Реална прогноза, която е преоразмерена към оригиналните данни.
		 */
		double[] forecast = new double[LEAD_LENGTH + 2];
		network.compute(normalizer.process(Arrays.copyOfRange(dst, dst.length - LAG_LENGTH, dst.length)), forecast);
		forecast[forecast.length - 2] = -Double.MAX_VALUE;
		forecast[forecast.length - 1] = +Double.MAX_VALUE;
		ACTIVATION.activationFunction(forecast, forecast.length - 2, 2);
		normalizer.setNormalizedLow(Arrays.stream(dst).min().getAsDouble());
		normalizer.setNormalizedHigh(Arrays.stream(dst).max().getAsDouble());
		forecast = Arrays.copyOfRange(normalizer.process(forecast), 0, forecast.length - 2);
		System.out.println("Forecast:\t" + Arrays.toString(forecast));
	}
}
