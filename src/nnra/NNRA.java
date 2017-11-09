package nnra;

import java.util.ArrayList;
import java.util.List;

public class NNRA {
	private double[][] similar; 	// 文档相似度矩阵
	private double[][] groupE; 		// 集团关联矩阵
	private double[][] matrixA; 	// 文档的隶属度矩阵
	private double alpha; 			// 阈值参数α，暂无作用
	private String[] textName;		// 文本名称
	private int[] tags;				// 标签数组
	private Integer[][] classesSet;	// 分类集合
	private int Nclasses;			// 类别数量
	private double tot;				// 相似度总和

	public NNRA(String[] textName, int[] tags, double[][] simialr, int Nclasses) {
		this.textName = textName;
		this.tags = tags;
		this.similar = simialr;
		this.Nclasses = Nclasses;
	}
	
	public NNRA(String[] textName, Integer[] tags, double[][] simialr, int Nclasses) {
		this.textName = textName;
		this.similar = simialr;
		this.Nclasses = Nclasses;
		this.tags = new int[tags.length];
		for (int i = 0; i < tags.length; i++) {
			this.tags[i] = tags[i];
		}
	}

	/**
	 * 开始执行算法
	 */
	public void excute() {
		// 初始化
		initClassesSet();
		calculateTot();
		initGroupE();
		initMatrxA();
		int length = tags.length;
		int[] newTags = new int[length];
		for (int i = 0; i < length; i++) {
			newTags[i] = tags[i];
		}
		List<int[]> candidate;
		while (true) {
			candidate = new ArrayList<>();
			// 找到更改后模块度变化量大于0的方案，返回队列
			candidate = getCandidate(candidate);
			// 若队列空，则无修改方案，可以结束算法
			if (candidate.size() == 0) {
				tags = newTags;
				return;
			}
			// 根据队列修改tags标签
			newTags = setCategory(candidate);
		}
	}

	/**
	 * 由类别标签集，得出文本名称所属类别集合，以下标的形式进行标记存储
	 */
	public void initClassesSet() {
		classesSet = new Integer[Nclasses][];
		List<Integer>[] tmp = new List[Nclasses];
		for (int i = 0; i < tags.length; i++) {
			int tag = tags[i];
			if (tmp[tag] == null) {
				tmp[tag] = new ArrayList<>();
			}
			tmp[tag].add(i);
		}
		for (int i = 0; i < Nclasses; i++) {
			classesSet[i] = tmp[i].toArray(
					new Integer[tmp[i].size()]);
		}
	}

	/**
	 * 计算tot，所有相似度总和
	 */
	public void calculateTot() {
		tot = 0.0;
		for (double[] row : similar) {
			for (double cell : row) {
				tot += cell;
			}
		}
	}

	/**
	 * groupE，集团关联矩阵，表示集团之间的关联强度
	 * 由相似度矩阵和tot计算得出
	 */
	public void initGroupE() {
		groupE = new double[Nclasses][Nclasses];
		for (int i = 0; i < groupE.length; i++) {
			for (int j = 0; j < groupE[i].length; j++) {
				groupE[i][j] = calE(i, j);
			}
		}
	}

	/**
	 * 计算groupE集团关联矩阵中的元素值，
	 * 参数为两个类别的标签，可以相同
	 * @param row 类别1
	 * @param col 类别2，可以与类别1相同
	 * @return	两个集团的关联强度
	 */
	public double calE(int row, int col) {
		double si = 0.0;
		for (Integer d1 : classesSet[row]) {
			for (Integer d2 : classesSet[col]) {
				si += similar[d1][d2];
			}
		}
		return si / tot;
	}

	/**
	 * MatrixA，文本隶属度行列式
	 * 行数为文本数量，列数为集团(类别)数量
	 * 表示某一本文对某一集团(类别)的隶属度
	 */
	public void initMatrxA() {
		matrixA = new double[tags.length][Nclasses];
		for (int i = 0; i < matrixA.length; i++) {
			for (int j = 0; j < Nclasses; j++) {
				double si = 0.0;
				for (int cl : classesSet[j]) {
					si += similar[i][cl];
				}
				matrixA[i][j] = si / tot;
			}
		}
	}

	/**
	 * 计算模块度Q
	 * @return 当前分类状态的模块度Q
	 */
	public double calculateModuleVal() {
		double moduleVal = 0.0;
		double rowTotalSquare = 0.0;
		for (int i = 0; i < Nclasses; i++) {
			double[] row = groupE[i];
			for (double cell : row) {
				rowTotalSquare += cell;
			}
			rowTotalSquare = rowTotalSquare * rowTotalSquare;
			moduleVal += (row[i] - rowTotalSquare);
		}
		return moduleVal;
	}

	/**
	 * 试调整分类，并计算出模块度变化量δQ
	 * 若δQ>0，把这种调整记录进列表当中，代表待调整的队列
	 * @param candidate 待调整的队列
	 * @return	计算后的待调整队列
	 */
	public List<int[]> getCandidate(List<int[]> candidate) {
		for (int i = 0; i < tags.length; i++) {
			int maxSubjectionTag = getMaxSubjectionClass(matrixA[i]);
			if (maxSubjectionTag != tags[i]) {
				double deltaModuleVal = calDeltaModuleVal(i, tags[i], maxSubjectionTag);
				if (deltaModuleVal > 0) {
					int[] tuple = { i, tags[i], maxSubjectionTag };
					candidate.add(tuple);
				}
			}
		}
		return candidate;
	}

	/**
	 * 找到指定文本的最大隶属度的类别
	 * @param allSubjection 该文本的所有隶属度
	 * @return 隶属度最大的类别下标
	 */
	public int getMaxSubjectionClass(double[] allSubjection) {
		double maxSubjection = -999999.9;
		int index = -1;
		for (int i = 0; i < allSubjection.length; i++) {
			if (allSubjection[i] > maxSubjection) {
				maxSubjection = allSubjection[i];
				index = i;
			}
		}
		return index;
	}

	/**
	 * 计算模块度变化量，仅与变化的新旧类别有关
	 * @param docIndex 文本下标，便于查找文本的相似度和隶属度
	 * @param oldTag 旧的分类标签
	 * @param newTag 新的分类标签
	 * @return
	 */
	public double calDeltaModuleVal(int docIndex, int oldTag, int newTag) {
		double deltaModuleVal = 0.0;
		double Dnew = matrixA[docIndex][newTag];
		double Dold = matrixA[docIndex][oldTag];
		double Dall = 0.0;
		double aNew = 0.0;
		double aOld = 0.0;
		for (double cell : groupE[oldTag]) {
			aOld += cell;
		}
		for (double cell : groupE[newTag]) {
			aNew += cell;
		}
		for (double cell : matrixA[docIndex]) {
			Dall += cell;
		}
		deltaModuleVal = (Dnew - Dold) - (Dall * (aNew - aOld) + Dall * Dall / 2);
		return deltaModuleVal;
	}

	/**
	 * 根据修改队列，作出具体的修改
	 * @param candidate 修改方案的队列
	 * @return	修改后的tags标签数组
	 */
	public int[] setCategory(List<int[]> candidate) {
		// TODO: 待添加α的作用
		int[] tags = this.tags;
		for (int[] tuple : candidate) {
			int index = tuple[0];
			int oldTag = tuple[1];
			int newTag = tuple[2];
			tags[index] = newTag;
			updateClassesSet(tags);
			updateMatrixA();
			updateGroupE(index, oldTag, newTag);
		}
		return tags;
	}

	/**
	 * 更新分类集
	 * @param tags
	 */
	public void updateClassesSet(int[] tags) {
		classesSet = new Integer[Nclasses][];
		List<Integer>[] tmp = new List[Nclasses];
		for (int i = 0; i < tags.length; i++) {
			int tag = tags[i];
			if (tmp[tag] == null) {
				tmp[tag] = new ArrayList<>();
			}
			tmp[tag].add(i);
		}
		for (int i = 0; i < Nclasses; i++) {
			if (tmp[i] != null) {
				classesSet[i] = tmp[i].toArray(new Integer[tmp[i].size()]);
			} else {
				classesSet[i] = new Integer[0];
			}
		}
	}

	/**
	 * 调整分类后
	 * 更新隶属度矩阵
	 */
	public void updateMatrixA() {
		for (int i = 0; i < matrixA.length; i++) {
			for (int j = 0; j < Nclasses; j++) {
				double si = 0.0;
				for (int cl : classesSet[j]) {
					si += similar[i][cl];
				}
				matrixA[i][j] = si / tot;
			}
		}
	}

	/**
	 * 调整分类后，更新集团关联强度矩阵GroupE
	 * @param docIndex 文本下标，便于找到隶属度和相似度
	 * @param oldTag 旧的分类标签
	 * @param newTag 新的分类标签
	 */
	public void updateGroupE(int docIndex, int oldTag, int newTag) {
		for (int i = 0; i < groupE.length; i++) {
			for (int j = 0; j < groupE[i].length; j++) {
				if (i == oldTag && j == oldTag) {
					groupE[i][j] = groupE[i][j] - matrixA[docIndex][oldTag];
				} else if (i == newTag && j == newTag) {
					groupE[i][j] = groupE[i][j] + matrixA[docIndex][newTag];
				} else if (i == oldTag && j == newTag || i == newTag && j == oldTag) {
					groupE[i][j] = groupE[i][j] + (matrixA[docIndex][oldTag] - matrixA[docIndex][newTag]) / 2;
				} else if (i == oldTag) {
					groupE[i][j] = groupE[i][j] - matrixA[docIndex][j] / 2;
				} else if (j == oldTag) {
					groupE[i][j] = groupE[i][j] - matrixA[docIndex][i] / 2;
				} else if (i == newTag) {
					groupE[i][j] = groupE[i][j] + matrixA[docIndex][j] / 2;
				} else if (j == newTag) {
					groupE[i][j] = groupE[i][j] + matrixA[docIndex][i] / 2;
				}
			}
		}
	}

	/**
	 * 输出groupE矩阵
	 */
	public void showGroupE() {
		for (double[] row : groupE) {
			for (double cell : row) {
				System.out.print(cell + "\t");
			}
			System.out.println();
		}
		System.out.println("end");
	}

	/**
	 * 输出MatrixA行列式
	 */
	public void showMatrixA() {
		for (double[] row : matrixA) {
			for (double cell : row) {
				System.out.print(cell + "\t");
			}
			System.out.println();
		}
		System.out.println("end");
	}

	/**
	 * 输出文本名称和分类
	 */
	public void showNameNTags() {
		for (int i = 0; i < tags.length; i++) {
			System.out.println(textName[i] + "\t" + tags[i]);
		}
	}
	
	public int[] getTags() {
		return this.tags;
	}

	/**
	 * 测试
	 * @param args
	 */
	public static void main(String[] args) {
		String[] testNameStrings = { "ID1", "ID2", "ID3", "ID4", "ID5", "ID6" };
		double[][] similar = { 	{ 1, 0.5, 0.4, 0.2, 0.6, 0.2 }, 
								{ 0.5, 1, 0.3, 0.4, 0.6, 0.3 }, 
								{ 0.4, 0.3, 1, 0.8, 0.1, 0.5 }, 
								{ 0.2, 0.4, 0.8, 1, 0.1, 0.4 }, 
								{ 0.6, 0.6, 0.1, 0.1, 1, 0.2 },
								{ 0.2, 0.3, 0.5, 0.4, 0.2, 1 } };
		int[] tags = { 0, 0, 1, 1, 1, 2 };
		NNRA testNnra = new NNRA(testNameStrings, tags, similar, 3);
		testNnra.excute();
		// testNnra.showGroupE();
		// testNnra.showMatrixA();
		testNnra.showNameNTags();
	}

}
