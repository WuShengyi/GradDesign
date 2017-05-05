#extracting useful information from howNet.txt,extracting prime meaning element
use Encode
open(file_in,"HowNet.txt") || die("open failed");
open(file_out,">word2mean.txt") || die("open falied");
$isWord=0;
$word="";
%Word_Mean=();
while(<file_in>){
	chomp();
	if($_=~/W_C=(\S.+)/){
		$isWord=1;
		$word=$1;
	}
	if($isWord==1){
		if($_=~/DEF={(\S.+)}/){
			$def_cont=$1;
			@conts=split(":",$def_cont);
			$prim_mean=$conts[0];
			push(@{$Word_Mean{$word}},$prim_mean);
			#print $prim_mean;
		}
	}
	if($_ eq "\n"){
		$isWord=0;
		$word="";
	}
}
foreach (keys %Word_Mean){	
	$means=$Word_Mean{$_};
	print file_out "$_ ";
	foreach (@{$means}){
		print file_out "$_ ";
	}
	print file_out "\n";
}
print "ok ";
close(file_in);
close(file_out);