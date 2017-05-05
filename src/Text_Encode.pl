#this is a perl program to Encode corpus file "ret.txt" to utf-8 format
use Encode;
open(file_in,$ARGV[0]) || die("fail to open source file");
open(file_out,$ARGV[1]) || die('failed to create output file');
while(<file_in>){
	chomp($_);
	print $_;
 	$_=decode('gbk',$_);
 	print $_;
 	print file_out $_;
 	#getc;
}
close(file_in);
close(file_out);	