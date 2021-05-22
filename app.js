var express = require('express');
var session = require('express-session');
var bodyParser = require('body-parser');
var app=express();


app.set('view engine', 'ejs');
app.use('/css',express.static('css'));
app.use('/images',express.static('images'));
app.use('/js',express.static('js'));
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({extended : true}));


app.get('/',function(req,res){
	res.render("Home");
	//console.log(req.user);
});

//Login
//Login Logic
app.post('/login',function(req,res){
    
    username = req.body.uname;
    passwd = req.body.pass;
    if(username == "admin" && passwd == "1234"){
        res.render('index');
    }
    else{
        res.render('home');
    }
});

//Result Post
app.post('/result',function(req,res){
    values= (req.body);
    var inputData = {'HR':req.body.heartrate,'O2Sat':req.body.o2sat,'temp':req.body.temp,
    'SBP':req.body.sbpbp,'MAP':req.body.map,'DBP':req.body.dbp,'Resp':req.body.resp,
    'FiO2':req.body.fio2,'pH':req.body.ph,'SaO2':req.body.sao2,'BUN':req.body.bun,
    'Calcium':req.body.calc,'Chloride':req.body.chloride,'Glucose':req.body.gluc,'Hgb':req.body.hgb,
    'WBC':req.body.wbc,'Age':req.body.age,'Gender':req.body.gender,'Unit1':req.body.unit1,
    'Unit2':req.body.unit2,'HospAdmTime':req.body.hosptime,'ICULOS':req.body.iculos}
    //var inputDataStr = JSON.stringify(inputData)
    //Call To python process
    const spawn = require("child_process").spawn;
    const pythonProcess = spawn('python',["sepsis_userinp_shub.py", inputData['HR'],inputData['O2Sat'],inputData['temp'],
    inputData['SBP'],inputData['MAP'],inputData['DBP'],inputData['Resp'],inputData['FiO2'],inputData['pH'],
    inputData['SaO2'],inputData['BUN'],inputData['Calcium'],inputData['Chloride'],inputData['Glucose'],inputData['Hgb'],
    inputData['WBC'],inputData['Age'],inputData['Gender'],inputData['Unit1'],inputData['Unit2'],inputData['HospAdmTime'],
    inputData['ICULOS']]);
    pythonProcess.stdout.on('data', (data) => {
        sepsisData = (data.toString())
        sepsisOutput= {'detect':sepsisData[0], 'level':sepsisData[3],'medicine' : JSON.parse(sepsisData.substr(6))}
        switch(sepsisOutput.detect){
            case "0":
                sepsisOutput.detect ="No"
                break;
            case "1":
                sepsisOutput.detect = "Yes"
                break;
        }
        switch(sepsisOutput.level){
            case "1":
                sepsisOutput.level = "Low "
                break;
            case "2":
                sepsisOutput.level = "Medium"
                break;
            case "3":
                sepsisOutput.level = "High"
                break;
        }
        var indexArr = [];

        // build the index
        for (var x in sepsisOutput.medicine) {
            indexArr.push(x);
        }
        console.log(sepsisOutput.medicine[indexArr[1]])
        medicine_list = ""
        var x;
        var count =1;
        for(x in sepsisOutput.medicine){
            medicine_list += String(count)+ "." + x + "\n"
            medicine_list+= sepsisOutput.medicine[x]+ `\n`
            count +=1
        }
        console.log(medicine_list)
        sepsisOutput.medicine = medicine_list
        //console.log(sepsisOutput)
        res.render("result",{sepsisOutput:sepsisOutput})
        //res.send(data.toString())
        res.end('end');
    });
    //res.render("result")
});


//process.env.PORT
app.listen(/*process.env.PORT*/2500,function(){
    console.log("Welcome to Sepsis System");
});