open(file_in,"$ARGV[0]") || die("open failed");
open(file_out,">$ARGV[1]")|| die("open failed");
$threshold=$ARGV[2];
while(<file_in>){
	chomp();
	@items=split(" ",$_);
	$newstr="";
	$count=0;
	while($count<=$threshold && $count<@items){
		$newstr.=$items[$count];
		$count++;
		$newstr.=" ";
	}
	print file_out "$newstr\n";
}
close(file_out);
close(file_in);
