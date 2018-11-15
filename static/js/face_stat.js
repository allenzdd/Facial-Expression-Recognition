function plot_bar_chart(json_data) {
    var myChart = echarts.init(document.getElementById('face_chart'));

    var option = {

        tooltip: {},
        legend: {
            data:['emtion']
        },
        xAxis: {
            data: ["angry","disgust","scared","happy","sad","surprised","neutral"]
        },
        yAxis: {},
        series: [{
            name: 'emotion',
            type: 'bar',
            data: [json_data['angry'],
                json_data['disgust'],
                json_data['scared'],
                json_data['happy'],
                json_data['sad'],
                json_data['surprised'],
                json_data['neutral']]
        }]
    };

    myChart.setOption(option);
}


function add_time_list(time_lst) {
    for (var i=0; i<time_lst.length; i++){
        console.log(time_lst[i]);
        document.write(`${time_lst[i][0]}-${time_lst[i][1]}&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp`);
        if ((i+1) % 3 === 0){
            // language=DjangoTemplate
            document.write('<br>');
        }
    }
}