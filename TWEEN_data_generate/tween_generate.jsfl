var source_folder = "C:\\MNIST_data";
var output_folder = "C:\\TWEEN_data";
// var clazz = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
var clazz = [8];
var morphing_start_frame = 0;
var morphing_end_frame = 99;

Date.prototype.Format = function (fmt) {
    var o = {
        "M+": this.getMonth() + 1, 
        "d+": this.getDate(),
        "h+": this.getHours(), 
        "m+": this.getMinutes(),
        "s+": this.getSeconds(),
        "q+": Math.floor((this.getMonth() + 3) / 3),
        "S": this.getMilliseconds()
    };
    if (/(y+)/.test(fmt)) fmt = fmt.replace(RegExp.$1, (this.getFullYear() + "").substr(4 - RegExp.$1.length));
    for (var k in o)
    if (new RegExp("(" + k + ")").test(fmt)) fmt = fmt.replace(RegExp.$1, (RegExp.$1.length == 1) ? (o[k]) : (("00" + o[k]).substr(("" + o[k]).length)));
    return fmt;
}

var output_sub_folder = output_folder + "\\" + new Date().Format("yyyy-MM-dd hh-mm-ss");
FLfile.createFolder(FLfile.platformPathToURI(output_sub_folder));

fl.trace("---------- OUTPUT FOLDER ----------");
fl.trace("path : " + FLfile.platformPathToURI(output_sub_folder));

// image selection start
var start_images_uri = [];
var end_images_uri = [];
var start_images_name = [];
var end_images_name = [];

for(i = 0;i < clazz.length;i++){

	var search_pattern = source_folder + "\\" + clazz[i] + "_*.jpg";
	var search_pattern_uri = FLfile.platformPathToURI(search_pattern);
	var sample_list = FLfile.listFolder(search_pattern_uri,"files");

	while(true){
	
		var randomPickNumber1 = Math.floor(sample_list.length * Math.random());
		var randomPickNumber2 = Math.floor(sample_list.length * Math.random());
	
		if(randomPickNumber2 != randomPickNumber1){
			break;
		}else{
			continue;
		}
	}
	
	start_images_name[i] = sample_list[randomPickNumber1];
	start_images_uri[i] = FLfile.platformPathToURI(source_folder + "\\" + sample_list[randomPickNumber1]);
	end_images_name[i] = sample_list[randomPickNumber2];
	end_images_uri[i] = FLfile.platformPathToURI(source_folder + "\\" + sample_list[randomPickNumber2]);
	
	fl.trace("---------- CLASS : " + clazz[i] + " ----------");
	fl.trace("from : " + start_images_uri[i]);
	fl.trace("to   : " + end_images_uri[i]);
	
};
// image selection end

// make morphing timeline start
var doc = fl.getDocumentDOM();

if( doc == null )
{
	doc = fl.createDocument("timeline");
}

var timeline = doc.getTimeline();
var layers = timeline.layers;

for(i = 0;i < start_images_name.length;i++){

	currentLayer = timeline.addNewLayer(clazz[i]);
	timeline.setSelectedLayers(currentLayer);
	
	doc.importFile(start_images_uri[i],true);
	doc.importFile(end_images_uri[i],true);
	
	timeline.convertToKeyframes(morphing_end_frame);

	timeline.setSelectedFrames(morphing_start_frame,morphing_start_frame);
	doc.library.addItemToDocument({x:14, y:14},start_images_name[i]);

	doc.selection = [timeline.layers[currentLayer].frames[morphing_start_frame].elements[0]];
	doc.traceBitmap(100, 8, 'pixels', 'normal');
	
	timeline.setSelectedFrames(morphing_end_frame,morphing_end_frame);
	doc.library.addItemToDocument({x:14, y:14},end_images_name[i]);

	doc.selection = [timeline.layers[currentLayer].frames[morphing_end_frame].elements[0]];
	doc.traceBitmap(100, 8, 'pixels', 'normal');

	timeline.layers[currentLayer].frames[morphing_start_frame].tweenType = "shape";
	
	var clazz_folder = output_sub_folder + "\\" + clazz[i];
	
	FLfile.createFolder(FLfile.platformPathToURI(clazz_folder));
	doc.exportPNG(FLfile.platformPathToURI(clazz_folder + "\\" + clazz[i] + "_.png"),true,false);
	
	timeline.layers[currentLayer].visible = false;

}

doc.close(false)
// make morphing timeline end