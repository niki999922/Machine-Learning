import Matrix.Companion.macroFb
import Matrix.Companion.microFb
import java.util.*
import kotlin.math.abs

class Matrix(private var matrix: List<List<Int>>) {
    val size = matrix.size
    val ALL = matrix.mapIndexed { ind, _ -> P(ind) }.stream().reduce(0) { acc, it -> acc + it }

    companion object {
        const val EPS = 1E-10

        fun Matrix.Precision(ind: Int): Double {
            return TP(ind).toDouble().safeDiv((TP(ind).toDouble() + FP(ind).toDouble()))
        }

        fun Matrix.Recall(ind: Int): Double {
            return TP(ind).toDouble().safeDiv(TP(ind).toDouble() + FN(ind).toDouble())
        }

        fun Matrix.PrecisionW(): Double {
            var res = 0.0
            for (i in 0 until size) {
                res += (T(i).toDouble() * C(i).toDouble()).safeDiv(P(i).toDouble())
            }
            return res / ALL.toDouble()
        }

        fun Matrix.RecallW(): Double {
            var res = 0
            for (i in 0 until size) {
                res += T(i)
            }
            return res.toDouble() / ALL.toDouble()
        }

        fun Matrix.FN(ind: Int): Int {
            var res = 0
            for (i in 0 until size) {
                if (ind != i) {
                    res += matrix[i][ind]
                }
            }
            return res
        }

        fun Matrix.FP(ind: Int): Int {
            return matrix[ind].filterIndexed { ind2, _ -> ind != ind2 }.stream().reduce(0) { acc, it -> acc + it }.toInt()
        }

        fun Matrix.TP(ind: Int): Int = T(ind)
        fun Matrix.T(ind: Int): Int = matrix[ind][ind]
        fun Matrix.P(ind: Int): Int = matrix[ind].stream().reduce(0) { acc, it -> acc + it }.toInt()
        fun Matrix.C(ind: Int): Int {
            var res = 0
            for (i in 0 until size) {
                res += matrix[i][ind]
            }
            return res
        }

        fun Matrix.F(ind: Int): Double {
            return 2.0 * (Precision(ind) * Recall(ind)).safeDiv((Precision(ind) + Recall(ind)))
        }

        fun Matrix.microFb(): Double {
            var res = 0.0
            for (i in 0 until size) {
                res += C(i).toDouble() * F(i)
            }
            return res / ALL.toDouble()
        }

        fun Matrix.macroFb(): Double {
            return 2.0 * (PrecisionW() * RecallW()) / (PrecisionW() + RecallW())
        }

        fun Double.safeDiv(b: Double): Double {
            return if (abs(this) < EPS) 0.0 else this / b

        }
    }
}

fun generateList(size: Int) = Collections.nCopies(size, 0).toMutableList()

fun main() {
    val k = readLine()!!.toInt()
    val tmp = mutableListOf<MutableList<Int>>()
    for (i in 0 until k) {
        tmp.add(generateList(k))
    }
    var tmpInd = 0
    repeat(k) {
        readLine()!!.split(" ").map(Integer::parseInt).forEachIndexed { ind,value ->
            tmp[tmpInd][ind] = value
        }
        tmpInd++
    }
    val matrix = Matrix(tmp)

    println(matrix.macroFb())
    println(matrix.microFb())
}