function mpld3_load_lib(url, callback){
  var s = document.createElement('script');
  s.src = url;
  s.async = true;
  s.onreadystatechange = s.onload = callback;
  s.onerror = function(){console.warn("failed to load library " + url);};
  document.getElementsByTagName("head")[0].appendChild(s);
}

var facecolors = ["#FF0000", "#B3FF00", "#FF00BD", "#07FF00", "#F7FF00", "#FF0000", "#00FF9D", "#000FFF", "#07FF00", "#FF0000", "#00FF9D", "#00FF9D", "#FF0000", "#FFA500", "#000FFF", "#07FF00", "#FFA500", "#FF0000", "#9B00FF", "#00B5FF", "#07FF00", "#00FF9D", "#07FF00", "#00FF9D", "#F7FF00", "#000FFF", "#00FF9D", "#FF0000", "#00B5FF", "#07FF00", "#FFA500", "#FF0000", "#9B00FF", "#B3FF00", "#9B00FF", "#FFA500", "#FF00BD", "#F7FF00", "#000FFF", "#FFA500", "#000FFF", "#FFA500", "#FF00BD", "#FFA500", "#B3FF00", "#F7FF00", "#FFA500", "#00FF9D", "#FF0000", "#07FF00", "#00FF9D", "#FF00BD", "#00B5FF", "#07FF00", "#FF00BD", "#FF0000", "#FF0000", "#FFA500", "#FF0000", "#FF0000", "#00B5FF", "#00B5FF", "#F7FF00", "#FF0000", "#F7FF00", "#F7FF00", "#00FF9D", "#00B5FF", "#000FFF", "#07FF00"];
var labels = ["A Christmas Carol in Prose; Being a Ghost Story of Christmas by Charles Dickens", "A Doll's House - a play by Henrik Ibsen", "A Study in Scarlet by Arthur Conan Doyle", "A Tale of Two Cities by Charles Dickens", "Adventures of Huckleberry Finn by Mark Twain", "Alice's Adventures in Wonderland by Lewis Carroll", "An Inquiry into the Nature and Causes of the Wealth of Nations by Adam Smith", "Anna Karenina by graf Leo Tolstoy", "Anne of Green Gables by L. M. Montgomery", "Around the World in Eighty Days by Jules Verne", "Autobiography of Benjamin Franklin by Benjamin Franklin", "Beyond Good and Evil by Friedrich Wilhelm Nietzsche", "Candide by Voltaire", "Common Sense by Thomas Paine", "Crime and Punishment by Fyodor Dostoyevsky", "David Copperfield by Charles Dickens", "Don Quixote by Miguel de Cervantes Saavedra", "Dracula by Bram Stoker", "Emma by Jane Austen", "Frankenstein; Or, The Modern Prometheus by Mary Wollstonecraft Shelley", "Great Expectations by Charles Dickens", "Il Principe", "Jane Eyre- An Autobiography by Charlotte Bronte\u0308", "John Locke___Second Treatise of Government", "Jonathan Swift___Gulliver's Travels", "Les Mise\u0301rables by Victor Hugo", "Leviathan by Thomas Hobbes", "Metamorphosis by Franz Kafka", "Moby Dick; Or, The Whale by Herman Melville", "Oliver Twist by Charles Dickens", "Paradise Lost by John Milton", "Peter Pan by J. M", "Pride and Prejudice by Jane Austen", "Pygmalion by Bernard Shaw", "Sense and Sensibility by Jane Austen", "Songs of Innocence, and Songs of Experience by William Blake", "The Adventures of Sherlock Holmes by Arthur Conan Doyle", "The Adventures of Tom Sawyer by Mark Twain", "The Brothers Karamazov by Fyodor Dostoyevsky", "The Complete Works of William Shakespeare by William Shakespeare", "The Count of Monte Cristo, Illustrated by Alexandre Dumas", "The Divine Comedy by Dante, Illustrated by Dante Alighieri", "The Hound of the Baskervilles by Arthur Conan Doyle", "The Iliad by Homer", "The Importance of Being Earnest: A Trivial Comedy for Serious People by Oscar Wilde", "The Jungle Book by Rudyard Kipling", "The King James Version of the Bible", "The Life and Adventures of Robinson Crusoe by Daniel Defoe", "The Mysterious Affair at Styles by Agatha Christie", "The Picture of Dorian Gray by Oscar Wilde", "The Republic by Plato", "The Return of Sherlock Holmes by Arthur Conan Doyle", "The Scarlet Letter by Nathaniel Hawthorne", "The Secret Adversary by Agatha Christie", "The Sign of the Four by Arthur Conan Doyle", "The Strange Case of Dr. Jekyll and Mr Hyde", "The Time Machine by H. G. Wells", "The Tragedy of Romeo and Juliet by William Shakespeare", "The War of the Worlds by H. G. Wells", "The Wonderful Wizard of Oz by L. Frank Baum", "The Works of Edgar Allan Poe \u2014 Volume 1 by Edgar Allan Poe", "The Works of Edgar Allan Poe \u2014 Volume 2 by Edgar Allan Poe", "Three Men in a Boat by Jerome K. Jerome", "Through the Looking-Glass by Lewis Carroll", "Treasure Island by Robert Louis Stevenson", "Ulysses by James Joyce", "Utopia by Saint Thomas More", "Walden, and On The Duty Of Civil Disobedience by Henry David Thoreau", "War and Peace by graf Leo Tolstoy", "Wuthering Heights by Emily Bronte\u0308"];

var headers = ["#FF00BD", "#F7FF00", "#000FFF", "#00FF9D", "#9B00FF", "#B3FF00", "#07FF00", "#FF0000", "#FFA500", "#00B5FF"];
for(var n = 1; n <= 2; n++)
{
  var table_data = {}
  var columns = headers.slice((n-1) * headers.length/2, (n-1) * headers.length/2 + headers.length/2)
  for(var i = 0; i < columns.length; i++)
  {
    table_data[columns[i]] = []
    $('#cluster_table_' + n + ' table thead tr').append('<th><div class="circle" style="background:' + columns[i] + '"></div></th>')
  }
  for(var i = 0; i < facecolors.length; i++)
    if(facecolors[i] in table_data)
      table_data[facecolors[i]].push(labels[i])
  var tbody = $('#cluster_table_' + n + ' table tbody')
  while(true)
  {
    var row = '<tr>';
    var present = false;
    for(var key in table_data)
    {
      if(i < table_data[key].length)
      {
        name = table_data[key][i];
        present = true;
      }
      else
        name = "";
      row += '<td>' + name + '</td>';
    }
    row += '</tr>';
    tbody.append(row);
    if(!present)
      break
  }
}
if(typeof(mpld3) !== "undefined" && mpld3._mpld3IsLoaded){
   // already loaded: just create the figure
   !function(mpld3){
mpld3.draw_figure("fig_el190494403413264275819626", {"axes": [{"xlim": [-1.5, 1.5], "yscale": "linear", "axesbg": "#FFFFFF", "texts": [], "zoomable": true, "images": [], "xdomain": [-1.5, 1.5], "ylim": [-1.5, 1.5], "paths": [], "sharey": [], "sharex": [], "axesbgalpha": null, "axes": [{"scale": "linear", "tickformat": null, "grid": {"gridOn": false}, "fontsize": 12.0, "position": "bottom", "nticks": 7, "tickvalues": null}, {"scale": "linear", "tickformat": null, "grid": {"gridOn": false}, "fontsize": 12.0, "position": "left", "nticks": 7, "tickvalues": null}], "lines": [], "markers": [], "id": "el190494403592272", "ydomain": [-1.5, 1.5], "collections": [{"paths": [[[[0.0, -0.5], [0.13260155, -0.5], [0.25978993539242673, -0.44731684579412084], [0.3535533905932738, -0.3535533905932738], [0.44731684579412084, -0.25978993539242673], [0.5, -0.13260155], [0.5, 0.0], [0.5, 0.13260155], [0.44731684579412084, 0.25978993539242673], [0.3535533905932738, 0.3535533905932738], [0.25978993539242673, 0.44731684579412084], [0.13260155, 0.5], [0.0, 0.5], [-0.13260155, 0.5], [-0.25978993539242673, 0.44731684579412084], [-0.3535533905932738, 0.3535533905932738], [-0.44731684579412084, 0.25978993539242673], [-0.5, 0.13260155], [-0.5, 0.0], [-0.5, -0.13260155], [-0.44731684579412084, -0.25978993539242673], [-0.3535533905932738, -0.3535533905932738], [-0.25978993539242673, -0.44731684579412084], [-0.13260155, -0.5], [0.0, -0.5]], ["M", "C", "C", "C", "C", "C", "C", "C", "C", "Z"]]], "edgecolors": ["#000000"], "edgewidths": [1.0], "offsets": "data01", "yindex": 1, "id": "el190494403734736", "pathtransforms": [[11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0]], "pathcoordinates": "display", "offsetcoordinates": "data", "zorder": 1, "xindex": 0, "alphas": [null], "facecolors": facecolors}], "xscale": "linear", "bbox": [0.125, 0.099999999999999978, 0.77500000000000002, 0.80000000000000004]}], "height": 480.0, "width": 640.0, "plugins": [{"type": "reset"}, {"enabled": false, "button": true, "type": "zoom"}, {"enabled": false, "button": true, "type": "boxzoom"}, {"voffset": 10, "labels": labels, "hoffset": 0, "location": "mouse", "type": "tooltip", "id": "el190494403734736"}], "data": {"data01": [[1.0190963391916017, -0.5891454497652846], [1.1820306985213536, 0.04719014442748133], [1.050944423671474, -0.5131342498077021], [-0.6203851950548745, -0.6119224864680887], [0.03356642688279251, -0.4582991226666515], [1.0085288914750539, -0.7206321982027825], [-0.26206102646014784, 1.0897692237167866], [-0.8957309351708628, -0.2928587464128543], [-0.08144679109054512, -0.8324071572316757], [0.07114826754334579, -0.22968316039380618], [-0.21392791941531436, 0.57343242273048], [-0.14170481178701635, 0.7894057366267917], [0.6983157233405062, 0.28688770139298364], [0.6884254105365228, 1.1198360779900622], [-0.7489698769655322, -0.4609477961573811], [-1.0582499775308314, -0.26197477168141564], [-0.7034879608295156, 0.47044886523513285], [0.3623968913833782, -0.7396277934666733], [-0.7002512252788219, 0.05402988690069535], [-0.24452149036834495, 0.2210199239190335], [-0.7292147036297024, -0.6411209025465356], [0.11911147197861353, 0.8330607849728502], [-0.9609126566564599, -0.37618814399190226], [0.020584665602839662, 1.1897301505974152], [-0.5958258021966278, 0.42154595853971955], [-0.8328379003475184, 0.12077198355306043], [-0.064646643695305, 0.9887969931159093], [1.0975507175867167, 0.13000063435244907], [-0.5581860790631168, -0.08172187132571943], [-0.8329622230233797, -0.7483163600233383], [0.25498663221860685, 0.9199332437843941], [1.3055587083848808, 0.3871170456854765], [-0.7681682870011232, 0.08869455938414647], [0.6899138620042052, -0.07746377604370192], [-0.6889697340928915, -0.03958210162381256], [0.34079087974490596, 1.0384333020910863], [-0.270616437561542, -0.758196345345586], [0.03452697279128119, -1.0024267189198584], [-1.02138520099406, -0.1600734650677131], [-0.28050859410440954, 0.8221232287133104], [-1.156575499851517, 0.20986671482305108], [-0.0025670531559703743, 0.868459157826183], [0.3294241706331961, -0.5378441585662955], [-0.16116901277939397, 0.712829070632223], [1.360537296465111, 0.19229372997478397], [0.6593479058162475, -0.20274956564574462], [-0.04356647443676097, 0.8078523776923495], [-0.4891658985786546, 0.523269518986482], [0.38408110807373863, -0.8023752118154975], [-0.055432193924974706, -0.6084515816642408], [-0.5985487142980902, 0.7370840846593172], [-0.29117273781055397, -0.9754874169450185], [-0.2651659967020398, 0.0886001079458045], [0.1270480671194956, -0.9216644555520462], [0.7362824505005875, -0.4382836998856114], [1.12382760636616, 0.14740618789259752], [0.9679622887186287, 0.05950269366677006], [0.8391640510241355, 0.7787012453442321], [0.17396512194587338, -0.3657149862727752], [0.42882169430339434, -0.5816183036964768], [-0.4814335879405179, -0.06392894958304099], [-0.3959220307807579, 0.12683118944156974], [0.3965781155070044, -0.6471477494121691], [0.9302951793262674, -0.656634993466046], [0.4291220247970574, -0.6794327792556287], [-0.2612978276685233, -0.2580472790014622], [0.08852781838527896, 1.108720707308529], [-0.14800192512601076, 0.27030404053018303], [-0.7211209900260162, -0.0749454808700556], [-0.6063504664425243, -0.8138994656787513]]}, "id": "el190494403413264"});
   }(mpld3);
}else if(typeof define === "function" && define.amd){
   // require.js is available: use it to load d3/mpld3
   require.config({paths: {d3: "/assets/js/d3"}});
   require(["d3"], function(d3){
      window.d3 = d3;
      mpld3_load_lib("/assets/js/mpld3.js", function(){
         
         mpld3.draw_figure("fig_el190494403413264275819626", {"axes": [{"xlim": [-1.5, 1.5], "yscale": "linear", "axesbg": "#FFFFFF", "texts": [], "zoomable": true, "images": [], "xdomain": [-1.5, 1.5], "ylim": [-1.5, 1.5], "paths": [], "sharey": [], "sharex": [], "axesbgalpha": null, "axes": [{"scale": "linear", "tickformat": null, "grid": {"gridOn": false}, "fontsize": 12.0, "position": "bottom", "nticks": 7, "tickvalues": null}, {"scale": "linear", "tickformat": null, "grid": {"gridOn": false}, "fontsize": 12.0, "position": "left", "nticks": 7, "tickvalues": null}], "lines": [], "markers": [], "id": "el190494403592272", "ydomain": [-1.5, 1.5], "collections": [{"paths": [[[[0.0, -0.5], [0.13260155, -0.5], [0.25978993539242673, -0.44731684579412084], [0.3535533905932738, -0.3535533905932738], [0.44731684579412084, -0.25978993539242673], [0.5, -0.13260155], [0.5, 0.0], [0.5, 0.13260155], [0.44731684579412084, 0.25978993539242673], [0.3535533905932738, 0.3535533905932738], [0.25978993539242673, 0.44731684579412084], [0.13260155, 0.5], [0.0, 0.5], [-0.13260155, 0.5], [-0.25978993539242673, 0.44731684579412084], [-0.3535533905932738, 0.3535533905932738], [-0.44731684579412084, 0.25978993539242673], [-0.5, 0.13260155], [-0.5, 0.0], [-0.5, -0.13260155], [-0.44731684579412084, -0.25978993539242673], [-0.3535533905932738, -0.3535533905932738], [-0.25978993539242673, -0.44731684579412084], [-0.13260155, -0.5], [0.0, -0.5]], ["M", "C", "C", "C", "C", "C", "C", "C", "C", "Z"]]], "edgecolors": ["#000000"], "edgewidths": [1.0], "offsets": "data01", "yindex": 1, "id": "el190494403734736", "pathtransforms": [[11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0]], "pathcoordinates": "display", "offsetcoordinates": "data", "zorder": 1, "xindex": 0, "alphas": [null], "facecolors": facecolors}], "xscale": "linear", "bbox": [0.125, 0.099999999999999978, 0.77500000000000002, 0.80000000000000004]}], "height": 480.0, "width": 640.0, "plugins": [{"type": "reset"}, {"enabled": false, "button": true, "type": "zoom"}, {"enabled": false, "button": true, "type": "boxzoom"}, {"voffset": 10, "labels": labels, "hoffset": 0, "location": "mouse", "type": "tooltip", "id": "el190494403734736"}], "data": {"data01": [[1.0190963391916017, -0.5891454497652846], [1.1820306985213536, 0.04719014442748133], [1.050944423671474, -0.5131342498077021], [-0.6203851950548745, -0.6119224864680887], [0.03356642688279251, -0.4582991226666515], [1.0085288914750539, -0.7206321982027825], [-0.26206102646014784, 1.0897692237167866], [-0.8957309351708628, -0.2928587464128543], [-0.08144679109054512, -0.8324071572316757], [0.07114826754334579, -0.22968316039380618], [-0.21392791941531436, 0.57343242273048], [-0.14170481178701635, 0.7894057366267917], [0.6983157233405062, 0.28688770139298364], [0.6884254105365228, 1.1198360779900622], [-0.7489698769655322, -0.4609477961573811], [-1.0582499775308314, -0.26197477168141564], [-0.7034879608295156, 0.47044886523513285], [0.3623968913833782, -0.7396277934666733], [-0.7002512252788219, 0.05402988690069535], [-0.24452149036834495, 0.2210199239190335], [-0.7292147036297024, -0.6411209025465356], [0.11911147197861353, 0.8330607849728502], [-0.9609126566564599, -0.37618814399190226], [0.020584665602839662, 1.1897301505974152], [-0.5958258021966278, 0.42154595853971955], [-0.8328379003475184, 0.12077198355306043], [-0.064646643695305, 0.9887969931159093], [1.0975507175867167, 0.13000063435244907], [-0.5581860790631168, -0.08172187132571943], [-0.8329622230233797, -0.7483163600233383], [0.25498663221860685, 0.9199332437843941], [1.3055587083848808, 0.3871170456854765], [-0.7681682870011232, 0.08869455938414647], [0.6899138620042052, -0.07746377604370192], [-0.6889697340928915, -0.03958210162381256], [0.34079087974490596, 1.0384333020910863], [-0.270616437561542, -0.758196345345586], [0.03452697279128119, -1.0024267189198584], [-1.02138520099406, -0.1600734650677131], [-0.28050859410440954, 0.8221232287133104], [-1.156575499851517, 0.20986671482305108], [-0.0025670531559703743, 0.868459157826183], [0.3294241706331961, -0.5378441585662955], [-0.16116901277939397, 0.712829070632223], [1.360537296465111, 0.19229372997478397], [0.6593479058162475, -0.20274956564574462], [-0.04356647443676097, 0.8078523776923495], [-0.4891658985786546, 0.523269518986482], [0.38408110807373863, -0.8023752118154975], [-0.055432193924974706, -0.6084515816642408], [-0.5985487142980902, 0.7370840846593172], [-0.29117273781055397, -0.9754874169450185], [-0.2651659967020398, 0.0886001079458045], [0.1270480671194956, -0.9216644555520462], [0.7362824505005875, -0.4382836998856114], [1.12382760636616, 0.14740618789259752], [0.9679622887186287, 0.05950269366677006], [0.8391640510241355, 0.7787012453442321], [0.17396512194587338, -0.3657149862727752], [0.42882169430339434, -0.5816183036964768], [-0.4814335879405179, -0.06392894958304099], [-0.3959220307807579, 0.12683118944156974], [0.3965781155070044, -0.6471477494121691], [0.9302951793262674, -0.656634993466046], [0.4291220247970574, -0.6794327792556287], [-0.2612978276685233, -0.2580472790014622], [0.08852781838527896, 1.108720707308529], [-0.14800192512601076, 0.27030404053018303], [-0.7211209900260162, -0.0749454808700556], [-0.6063504664425243, -0.8138994656787513]]}, "id": "el190494403413264"});
      });
    });
}else{
    // require.js not available: dynamically load d3 & mpld3
    mpld3_load_lib("/assets/js/d3.js", function(){
         mpld3_load_lib("/assets/js/mpld3.js", function(){
                 
                 mpld3.draw_figure("fig_el190494403413264275819626", {"axes": [{"xlim": [-1.5, 1.5], "yscale": "linear", "axesbg": "#FFFFFF", "texts": [], "zoomable": true, "images": [], "xdomain": [-1.5, 1.5], "ylim": [-1.5, 1.5], "paths": [], "sharey": [], "sharex": [], "axesbgalpha": null, "axes": [{"scale": "linear", "tickformat": null, "grid": {"gridOn": false}, "fontsize": 12.0, "position": "bottom", "nticks": 7, "tickvalues": null}, {"scale": "linear", "tickformat": null, "grid": {"gridOn": false}, "fontsize": 12.0, "position": "left", "nticks": 7, "tickvalues": null}], "lines": [], "markers": [], "id": "el190494403592272", "ydomain": [-1.5, 1.5], "collections": [{"paths": [[[[0.0, -0.5], [0.13260155, -0.5], [0.25978993539242673, -0.44731684579412084], [0.3535533905932738, -0.3535533905932738], [0.44731684579412084, -0.25978993539242673], [0.5, -0.13260155], [0.5, 0.0], [0.5, 0.13260155], [0.44731684579412084, 0.25978993539242673], [0.3535533905932738, 0.3535533905932738], [0.25978993539242673, 0.44731684579412084], [0.13260155, 0.5], [0.0, 0.5], [-0.13260155, 0.5], [-0.25978993539242673, 0.44731684579412084], [-0.3535533905932738, 0.3535533905932738], [-0.44731684579412084, 0.25978993539242673], [-0.5, 0.13260155], [-0.5, 0.0], [-0.5, -0.13260155], [-0.44731684579412084, -0.25978993539242673], [-0.3535533905932738, -0.3535533905932738], [-0.25978993539242673, -0.44731684579412084], [-0.13260155, -0.5], [0.0, -0.5]], ["M", "C", "C", "C", "C", "C", "C", "C", "C", "Z"]]], "edgecolors": ["#000000"], "edgewidths": [1.0], "offsets": "data01", "yindex": 1, "id": "el190494403734736", "pathtransforms": [[11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0], [11.11111111111111, 0.0, 0.0, 11.11111111111111, 0.0, 0.0]], "pathcoordinates": "display", "offsetcoordinates": "data", "zorder": 1, "xindex": 0, "alphas": [null], "facecolors": facecolors}], "xscale": "linear", "bbox": [0.125, 0.099999999999999978, 0.77500000000000002, 0.80000000000000004]}], "height": 480.0, "width": 640.0, "plugins": [{"type": "reset"}, {"enabled": false, "button": true, "type": "zoom"}, {"enabled": false, "button": true, "type": "boxzoom"}, {"voffset": 10, "labels": labels, "hoffset": 0, "location": "mouse", "type": "tooltip", "id": "el190494403734736"}], "data": {"data01": [[1.0190963391916017, -0.5891454497652846], [1.1820306985213536, 0.04719014442748133], [1.050944423671474, -0.5131342498077021], [-0.6203851950548745, -0.6119224864680887], [0.03356642688279251, -0.4582991226666515], [1.0085288914750539, -0.7206321982027825], [-0.26206102646014784, 1.0897692237167866], [-0.8957309351708628, -0.2928587464128543], [-0.08144679109054512, -0.8324071572316757], [0.07114826754334579, -0.22968316039380618], [-0.21392791941531436, 0.57343242273048], [-0.14170481178701635, 0.7894057366267917], [0.6983157233405062, 0.28688770139298364], [0.6884254105365228, 1.1198360779900622], [-0.7489698769655322, -0.4609477961573811], [-1.0582499775308314, -0.26197477168141564], [-0.7034879608295156, 0.47044886523513285], [0.3623968913833782, -0.7396277934666733], [-0.7002512252788219, 0.05402988690069535], [-0.24452149036834495, 0.2210199239190335], [-0.7292147036297024, -0.6411209025465356], [0.11911147197861353, 0.8330607849728502], [-0.9609126566564599, -0.37618814399190226], [0.020584665602839662, 1.1897301505974152], [-0.5958258021966278, 0.42154595853971955], [-0.8328379003475184, 0.12077198355306043], [-0.064646643695305, 0.9887969931159093], [1.0975507175867167, 0.13000063435244907], [-0.5581860790631168, -0.08172187132571943], [-0.8329622230233797, -0.7483163600233383], [0.25498663221860685, 0.9199332437843941], [1.3055587083848808, 0.3871170456854765], [-0.7681682870011232, 0.08869455938414647], [0.6899138620042052, -0.07746377604370192], [-0.6889697340928915, -0.03958210162381256], [0.34079087974490596, 1.0384333020910863], [-0.270616437561542, -0.758196345345586], [0.03452697279128119, -1.0024267189198584], [-1.02138520099406, -0.1600734650677131], [-0.28050859410440954, 0.8221232287133104], [-1.156575499851517, 0.20986671482305108], [-0.0025670531559703743, 0.868459157826183], [0.3294241706331961, -0.5378441585662955], [-0.16116901277939397, 0.712829070632223], [1.360537296465111, 0.19229372997478397], [0.6593479058162475, -0.20274956564574462], [-0.04356647443676097, 0.8078523776923495], [-0.4891658985786546, 0.523269518986482], [0.38408110807373863, -0.8023752118154975], [-0.055432193924974706, -0.6084515816642408], [-0.5985487142980902, 0.7370840846593172], [-0.29117273781055397, -0.9754874169450185], [-0.2651659967020398, 0.0886001079458045], [0.1270480671194956, -0.9216644555520462], [0.7362824505005875, -0.4382836998856114], [1.12382760636616, 0.14740618789259752], [0.9679622887186287, 0.05950269366677006], [0.8391640510241355, 0.7787012453442321], [0.17396512194587338, -0.3657149862727752], [0.42882169430339434, -0.5816183036964768], [-0.4814335879405179, -0.06392894958304099], [-0.3959220307807579, 0.12683118944156974], [0.3965781155070044, -0.6471477494121691], [0.9302951793262674, -0.656634993466046], [0.4291220247970574, -0.6794327792556287], [-0.2612978276685233, -0.2580472790014622], [0.08852781838527896, 1.108720707308529], [-0.14800192512601076, 0.27030404053018303], [-0.7211209900260162, -0.0749454808700556], [-0.6063504664425243, -0.8138994656787513]]}, "id": "el190494403413264"});
            })
         });
}