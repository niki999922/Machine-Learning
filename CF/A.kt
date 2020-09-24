fun main() {
    val (_, _, k) = readLine()!!.split(" ").map { it.toInt() }
    val map = mutableMapOf<Int, MutableList<Int>>()
    val res = mutableListOf<MutableList<Int>>()
    for (i in 0 until k) {
        res.add(i, mutableListOf())
    }
    readLine()!!.split(" ").map(Integer::parseInt).forEachIndexed{ index, it ->
        map.computeIfPresent(it - 1) { _,value ->
            value.add(index + 1)
            return@computeIfPresent value
        }
        map.putIfAbsent(it - 1, mutableListOf(index + 1))
    }
    var itter = 0
    map.forEach {(_, values) ->
        values.forEach {
            res[itter % k].add(it)
            itter++
        }
    }
    res.forEach {
        print("${it.size} ")
        it.forEach {
            print("$it ")
        }
        println("")
    }
}