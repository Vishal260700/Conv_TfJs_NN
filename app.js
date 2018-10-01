let data;

let date = [];

let year = [];
let month = [];
let day = [];

let x = [];

let yLow = [];
let yHigh = [];
let yVol = [];
let yClose = [];
let yOpen = []; // initially we are not going to normalize the data to see if acc is less than we will try normalizing...

let model;

let altx = [];


let tfxs;
let tfysOpen;
let tfysLow;
let tfysHigh;
let tfysVol;
let tfysClose;

function preload(){
	data = loadJSON('./data.json')
}

function setup(){

	// refining data...


	for(let record of data.entries) {
		let col = record.Date;

		date.push(col);

		altx.push(parseInt(col)); // alternate approach
	}

	console.log(altx);

	// console.log(date.length);

	for(var i = 0 ; i<date.length ; i++){
			let x1 = parseInt(date[i][0])*1000 + parseInt(date[i][1])*100 + parseInt(date[i][2])*10 + parseInt(date[i][3]);
			year.push(x1);
			let x2 = parseInt(date[i][5])*10 + parseInt(date[i][6]);
			month.push(x2);
			let x3 = parseInt(date[i][8])*10 + parseInt(date[i][9]);
			day.push(x3);
	}
	
	/*
	console.log(year);
	console.log(month);
	console.log(day);
	*/

	// making final array for processing the independent variable i.e. x..

	for(var i = 0 ; i < date.length ; i ++){
		let temp = [];
		temp.push(day[i] / 31);
		temp.push(month[i] / 12);
		temp.push((year[i] - 1980) / 38);
		x.push(temp);
	}

	// processing dependent variable y
	y_Open_processing();
	y_Close_processing();
	y_High_processing();
	y_Low_processing();
	y_Volume_processing();

	// console.log(yOpen,yClose,yHigh,yLow,yVol);

	// now we have our both x and y variables here y are a couple of dependent so we will have a generalized model to predict different amt of them

	// model architecture

	model = tf.sequential();

	let hidden1 = tf.layers.conv1d({
		 inputShape : [altx.length , 1],
		 kernelSize: 100,
	     filters: 8,
	     strides: 2,
	     activation: 'relu',
	     kernelInitializer: 'VarianceScaling'
	});

	let hidden1_sup = tf.layers.maxPooling1d({
      	poolSize: [500],
      	strides: [2]
    });

	let hidden2 = tf.layers.conv1d({
	    kernelSize: 5,
	    filters: 16,
	    strides: 1,
	    activation: 'relu',
	    kernelInitializer: 'VarianceScaling'
    });

    let hidden2_sup = tf.layers.maxPooling1d({
        poolSize: [100],
        strides: [2]
    });

    let output = tf.layers.dense({
        units: 10,
        kernelInitializer: 'VarianceScaling',
        activation: 'softmax'
    });

    model.add(hidden1);
    model.add(hidden1_sup);
    model.add(hidden2);
    model.add(hidden2_sup);
    model.add(output);

    model.compile({
    	optimizer: 'sgd', 
    	loss: 'binaryCrossentropy', 
    	lr: 0.1
    });

    // structuring complete

    // now we need to convert the array into tensors for processing

    tfysOpen = tf.tensor1d(yOpen);

    // not working
    // tfxs = tf.tensor2d( x,[1960 , 3]);

    console.log(altx);

    tfxs = tf.tensor1d(altx);

    tfxs.print();



    // lets fit the model with all that we have done...

    const ep = 5;

    const options = {
    	epochs : ep,
    	batchsize : 3,
    	shuffle : true
    };

    model.fit(tfxs.reshape([1,1960,1]),tfysOpen.reshape([1,1960,1]),options);

}

function y_Open_processing(){
	for(let record of data.entries) {
		let col = record.Open;
		yOpen.push(col);
	}
}

function y_High_processing(){
	for(let record of data.entries) {
		let col = record.High;
		yHigh.push(col);
	}
}

function y_Low_processing(){
	for(let record of data.entries) {
		let col = record.Low;
		yLow.push(col);
	}
}

function y_Close_processing(){
	for(let record of data.entries) {
		let col = record.Close;
		yClose.push(col);
	}
}

function y_Volume_processing(){
	for(let record of data.entries) {
		let col = record.Volume;
		yVol.push(col);
	}
}


// what else to do

// Get a seperate test data to predict 
// leaking tensors