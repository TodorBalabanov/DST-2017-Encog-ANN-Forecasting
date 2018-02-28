import java.util.Random;

import org.encog.engine.network.activation.ActivationFunction;
import org.encog.engine.network.activation.ActivationTANH;

public class Constants {
	public static final Random PRNG = new Random();
	
	public static final long NUMBER_OF_EXPERIMENTS = 30;
	
	/*
	 * ����� �������, � �����������, �� ����� �� ��������.
	 */
	public static final long MAX_TRAINING_MILLISECONDS = 10000;

	/*
	 * ��������, � �����������, ���� ����� �� �� ����� �������� �� ����������.
	 */
	public static final long SINGLE_MEASUREMENT_MILLISECONDS = 100;

	/*
	 * ����� �� ������������ �������. ��������� � ���� �� �� ���� � ������ �� ��
	 * ������ ���������� ������������ ������� � �� �� ���������� ���������� �������
	 * ��� ������� �� ��������, ������ �� ������������� �� �������������� �������.
	 */
	public final static ActivationFunction ACTIVATION = new ActivationTANH();
	
	public static final int MIN_HIDDEN_NUMBER = 1;

	public static final int MAX_HIDDEN_NUBMER = 10;

	public static final int MIN_HIDDEN_LENGTH = 1;

	public static final int MAX_HIDDEN_LENGTH = 100;
}
