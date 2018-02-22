import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

import org.encog.engine.network.activation.ActivationFunction;
import org.encog.engine.network.activation.ActivationTANH;

public class Main {

	/*
	 * Размер на прозореца в миналото, който се подава на входа на изкуствената
	 * невронна мрежа.
	 */
	private final static int LAG_LENGTH = 15;

	/*
	 * Размер на прозореца в бъдещето, който се очаква като прогноза от изкуствената
	 * невронна мрежа.
	 */
	private final static int LEAD_LENGTH = 5;

	/*
	 * Избор на активационна функция. Направено е така за да може с лекота да се
	 * сменят различните активационни функции и да се наблюдават евентуални разлики
	 * при процеса на обучение, идващи от особеносттите на активационната функция.
	 */
	private final static ActivationFunction ACTIVATION = new ActivationTANH();

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
		double range[] = { -Double.MAX_VALUE, Double.MAX_VALUE };
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
		double min = Arrays.stream(dst).min().getAsDouble();
		double max = Arrays.stream(dst).max().getAsDouble();
		double scaled[] = Arrays.copyOf(dst, dst.length);
		for (int i = 0; i < scaled.length; i++) {
			scaled[i] = (range[1] - range[0]) * (scaled[i] - min) / (max - min) + range[0];
		}

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

		System.err.println(min);
		System.err.println(max);
		System.err.println(Arrays.stream(scaled).min().getAsDouble());
		System.err.println(Arrays.stream(scaled).max().getAsDouble());
	}

}
