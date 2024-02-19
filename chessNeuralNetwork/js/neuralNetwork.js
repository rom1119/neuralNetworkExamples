class NeuralNetwork {

    LEARNING_RATE = 0.01
    EPOCH = 1000

    w1 = []
    b1 = []
    
    w2 = []
    b2 = []
    
    w3 = []
    b3 = []
    
    w4 = []
    b4 = []
    
    w5 = []
    b5 = []

    z1 = []
    a1 = []
    z2 = []
    a2 = []
    z3 = []
    a3 = []
    z4 = []
    a4 = []
    z5 = []
    a5 = []

    constructor() {

        var layerOne = this.initLayerRandom(100, 64)
        var layerTwo = this.initLayerRandom(200, 100)
        var layerThree = this.initLayerRandom(100, 200)
        var layerFour = this.initLayerRandom(64, 100)
        var layerFive = this.initLayerRandom(64, 64)

        this.w1 = layerOne.weight
        this.b1 = layerOne.bias
        this.w2 = layerTwo.weight
        this.b2 = layerTwo.bias
        this.w3 = layerThree.weight
        this.b3 = layerThree.bias
        this.w4 = layerFour.weight
        this.b4 = layerFour.bias
        this.w5 = layerFive.weight
        this.b5 = layerFive.bias

    }

    initLayerRandom(rows, columns) {
        
        var weight = []
        var bias = []

        for (let i = 0; i < rows; i++) {
            let row = []
            for (let m = 0; m < columns; m++) {
                var weightVal = ((Math.random()*2-1) / 10)
                row.push(weightVal)
            }
            var biasVal = ((Math.random()*2-1) / 10)
            bias.push([biasVal])
            weight.push(row)
        }

        return {
            weight: weight,
            bias: bias,
        }
    }


    forwardPropagation(X) {
        var res = {}
        // console.log('this.w1', this.w1);
        // console.log('X', X);
        var firstCalc = calculateLayer(this.w1, X, this.b1)
        // console.log('firstCalc', firstCalc.z);
        res.z1 = firstCalc.z
        res.a1 = firstCalc.a
        
        var secondCalc = calculateLayer(this.w2, res.a1, this.b2)
        res.a2 = secondCalc.a
        res.z2 = secondCalc.z
        
        var thirdCalc = calculateLayer(this.w3, res.a2, this.b3)
        res.a3 = thirdCalc.a
        res.z3 = thirdCalc.z
        
        var fourthCalc = calculateLayer(this.w4, res.a3, this.b4)
        res.a4 = fourthCalc.a
        res.z4 = fourthCalc.z
        
        var fifthCalc = calculateLayer(this.w5, res.a4, this.b5)
        res.a5 = fifthCalc.a
        res.z5 = fifthCalc.z

        return res
    }

    backwardPropagation(forwardResult, X, Y) {
        var res = {}

        var dz5 = calcMinus(forwardResult.a5, Y)

        res.dw5 = calcMultiple(dz5, forwardResult.a4)
        res.db5 = dz5

        var dz4 = calcMultiple(calcDotProducts(transpose(this.w5), dz5), (tanhDerrivMatrix(forwardResult.z4)))
        res.dw4 = calcDotProducts(dz4, transpose(forwardResult.a3))
        res.db4 = sumMatrix(dz4)

        var dz3 = calcMultiple(calcDotProducts(transpose(this.w4), dz4), (tanhDerrivMatrix(forwardResult.z3)))
        res.dw3 = calcDotProducts(dz3, transpose(forwardResult.a2))
        res.db3 = sumMatrix(dz3)
    
    
        var dz2 = calcMultiple(calcDotProducts(transpose(this.w3), dz3), (tanhDerrivMatrix(forwardResult.z2)))
        res.dw2 = calcDotProducts(dz2, transpose(forwardResult.a1))
        res.db2 = sumMatrix(dz2)
    
        var dz1 = calcMultiple(calcDotProducts(transpose(this.w2), dz2), (tanhDerrivMatrix(forwardResult.z1)))
        
        res.dw1 = calcDotProducts(dz1, transpose(X))
        res.db1 = sumMatrix(dz1)

        return res

    }

    predict(X) {
        var forwardProps = this.forwardPropagation(X)

        return forwardProps.a5
    }


    train(X, Y) {
        const start = Date.now();
        for (var i = 0; i < this.EPOCH; i++) {
            var forwardProps = this.forwardPropagation(X)
            this.z1 = forwardProps.z1
            this.a1 = forwardProps.a1
            this.z2 = forwardProps.z2
            this.a2 = forwardProps.a2
            this.z3 = forwardProps.z3
            this.a3 = forwardProps.a3
            this.z4 = forwardProps.z4
            this.a4 = forwardProps.a4
            this.z5 = forwardProps.z5
            this.a5 = forwardProps.a5
    
            var backwardProps = this.backwardPropagation(forwardProps, X, Y)

            this.w1 = calcMinus(this.w1, calcMultiple(backwardProps.dw1, this.LEARNING_RATE))
            this.w2 = calcMinus(this.w2, calcMultiple(backwardProps.dw2, this.LEARNING_RATE))
            this.w3 = calcMinus(this.w3, calcMultiple(backwardProps.dw3, this.LEARNING_RATE))
            this.w4 = calcMinus(this.w4, calcMultiple(backwardProps.dw4, this.LEARNING_RATE))
            this.w5 = calcMinus(this.w5, calcMultiple(backwardProps.dw5, this.LEARNING_RATE))
            this.b1 = calcMinus(this.b1, calcMultiple(backwardProps.db1, this.LEARNING_RATE))
            this.b2 = calcMinus(this.b2, calcMultiple(backwardProps.db2, this.LEARNING_RATE))
            this.b3 = calcMinus(this.b3, calcMultiple(backwardProps.db3, this.LEARNING_RATE))
            this.b4 = calcMinus(this.b4, calcMultiple(backwardProps.db4, this.LEARNING_RATE))
            this.b5 = calcMinus(this.b5, calcMultiple(backwardProps.db5, this.LEARNING_RATE))

            if (i % 100 == 0) {
                // console.log('EPOCH', i)

            }
        }
        const end = Date.now();
        console.log(`Execution time: ${end - start} ms`);


        // console.log('X', X)
        // console.log('Y', Y)
        // console.log('this.w1', this.w1)
        // console.log('this.b1', this.b1)

        // console.log('this.w2', this.w2)
        // console.log('this.b2', this.b2)
        
        // console.log('this.w3', this.w3)
        // console.log('this.b3', this.b3)
        
        // console.log('this.w4', this.w4)
        // console.log('this.b4', this.b4)
        
        // console.log('this.w5', this.w5)
        // console.log('this.b5', this.b5)
        // console.log('z1', z1)
        // console.log('a1', a1)
        
        // console.log('z2', z2)
        // console.log('a2', a2)
        
        // console.log('z3', z3)
        // console.log('a3', a3)
        
        // console.log('z4', z4)
        // console.log('a4', a4)
        
        // console.log('z5', z5)
        // console.log('a5', this.a5)
        // console.log((net.b1))
    }

}