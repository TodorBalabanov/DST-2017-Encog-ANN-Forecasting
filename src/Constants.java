import java.util.Random;

import org.encog.engine.network.activation.ActivationFunction;
import org.encog.engine.network.activation.ActivationTANH;

public class Constants {
	public static final Random PRNG = new Random();

	public static final long NUMBER_OF_EXPERIMENTS = 1;

	/*
	 * Горна граница, в милисекунди, за време на обучение.
	 */
	public static final long MAX_TRAINING_MILLISECONDS = 60000;

	/*
	 * Интервал, в милисекунди, през който ще се очита прогреса на обучението.
	 */
	public static final long SINGLE_MEASUREMENT_MILLISECONDS = 1000;

	/*
	 * Избор на активационна функция. Направено е така за да може с лекота да се
	 * сменят различните активационни функции и да се наблюдават евентуални разлики
	 * при процеса на обучение, идващи от особеносттите на активационната функция.
	 */
	public final static ActivationFunction ACTIVATION = new ActivationTANH();

	public static final int MIN_HIDDEN_NUMBER = 1;

	public static final int MAX_HIDDEN_NUBMER = 10;

	public static final int MIN_HIDDEN_LENGTH = 1;

	public static final int MAX_HIDDEN_LENGTH = 100;
}
