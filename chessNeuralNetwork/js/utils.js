
function calculateLayer(weight, byMultiple,  bias) {
    var lenRowWeight = weight.length
    var multipleArray = byMultiple
    if (typeof multipleArray == 'number') {
        multipleArray = Array.from({length: weight.length}, (_, i) => [byMultiple])
    }
    
    var a = []
    var z = []
    for (let i = 0; i < lenRowWeight; i++) {
        a.push([])
        z.push([])

        var row = weight[i]
        var lenColumnWeight = row.length
        var zNew = 0

        var currentCol = 0
        var totalCols = multipleArray[0].length
        while (currentCol < totalCols) {

            for (let j = 0; j < lenColumnWeight; j++) {
                zNew += byMultiple[j][currentCol] * row[j]      
            }
            if (bias[i].length > 1) {
                
                zNew += bias[i][j]
            } else {
                zNew += bias[i][0]
            }
            
            var aVal = this.tanh(zNew)
            a[i].push(aVal)
            z[i].push(zNew)
            currentCol = currentCol + 1
        }
    }

    return {
        a: a,
        z: z,
    }
}

function calcMultiple(layer, byMultiple){

    if ((typeof layer == 'number') && (typeof byMultiple == 'number')) {
        return layer * byMultiple
    }
    var z = []

    var multipleArray = byMultiple

    if (typeof multipleArray == 'number') {
        multipleArray = Array.from({ length: layer.length }, (_, i) => [byMultiple])

    } else if (byMultiple.length < layer.length) {
        multipleArray = Array.from({ length: layer.length }, (_, i) => byMultiple[0])
    }

    if (layer.length < multipleArray.length) {
        var layerTmp = layer
        var byMultiplTmp = byMultiple
        layer = byMultiplTmp
        byMultiple = Array.from({ length: byMultiple.length }, (_, i) => layerTmp[0])
    }
    
    var lenRowWeight = layer.length

    for (let keyRow = 0; keyRow < lenRowWeight; keyRow++) {
        var row = layer[keyRow]
        z.push([])
        var lenColumn = row.length
        for (let m = 0; m < lenColumn; m++) {
            var zNew = 0
            var w1Cell = row[m]
            if (Array.isArray(byMultiple)) {
                if (byMultiple[keyRow] != 'null' ) {
                    zNew = w1Cell * byMultiple[keyRow][m]
                } else {
                    zNew = w1Cell * byMultiple[keyRow][0]
                }
            } else {
                zNew = w1Cell * byMultiple
            }

            z[keyRow].push(zNew)
        }
    }
    return z
}

function calcDotProducts(layer, byMultiple) {
    var z = []
    var multipleArray = byMultiple
    if (typeof multipleArray == 'number') {
        multipleArray = Array.from({length: weight.length}, (_, i) => [byMultiple])
    }
    var lenRowWeight = layer.length

    for (let keyRow = 0; keyRow < lenRowWeight; keyRow++) {
        var row = layer[keyRow]
        z.push([])
        var currentCol = 0
        var totalCols = multipleArray[0].length

        var lenColumnWeight = row.length

        while (currentCol < totalCols) {
            var zNew = 0
            for (let j = 0; j < lenColumnWeight; j++) {
                // console.log(multipleArray);
                // console.log(j);
                // console.log(currentCol);
                zNew += multipleArray[j][currentCol] * row[j]      

            }
     
            z[keyRow].push(zNew)
            currentCol = currentCol + 1
        }
    }

    return z
}


function calcMinus(layer, byMinus) {
    if ((typeof layer == 'number') && (typeof byMinus == 'number')) {
        return layer - byMinus
    }
    z = []
    var lenRowWeight = layer.length

    for (let keyRow = 0; keyRow < lenRowWeight; keyRow++) {
        var row = layer[keyRow]
        z.push([])
        var lenColumn = row.length
        for (let m = 0; m < lenColumn; m++) {
            var zNew = 0
            var w1Cell = row[m]

            if (Array.isArray(byMinus)) {
                if (byMinus[keyRow].length > 1 ) {
                    zNew = w1Cell - byMinus[keyRow][m]
                } else {
                    zNew = w1Cell - byMinus[keyRow][0]
                }
            } else {
                zNew = w1Cell - byMinus
            }

            z[keyRow].push(zNew)
        }
    }
    return z
}


function transpose(mat) {
    return mat[0].map((_, colIndex) => mat.map(row => row[colIndex]));
}

function sumMatrix(matrix) {
    var lenRow = matrix.length
    var sumVal = 0

    for (let i = 0; i < lenRow; i++) {
        var row = matrix[i]
        var lenColumn = row.length
        for (let m = 0; m < lenColumn; m++) {
            sumVal += row[m]
        }
    }
    return sumVal
}


function tanh(x) {
    return Math.tanh(x)
}

function tanhDerrivMatrix(matrix) {
    var lenRow = matrix.length

    for (let i = 0; i < lenRow; i++) {
        var row = matrix[i]
        var lenColumn = row.length
        for (let m = 0; m < lenColumn; m++) {
            matrix[i][m] = 1 - (this.tanh(row[m]) ** 2)
        }
    }
    return matrix
}