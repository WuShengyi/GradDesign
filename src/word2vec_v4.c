//this is a word2vec in c
//下面是我对word2vec.c的注释  
//详细算法可以参考论文，或者看这篇博客 http://www.cnblogs.com/downtjs/p/3784440.html  
  
  
//  Copyright 2013 Google Inc. All Rights Reserved.  
//  
//  Licensed under the Apache License, Version 2.0 (the "License");  
//  you may not use this file except in compliance with the License.  
//  You may obtain a copy of the License at  
//  
//      http://www.apache.org/licenses/LICENSE-2.0  
//  
//  Unless required by applicable law or agreed to in writing, software  
//  distributed under the License is distributed on an "AS IS" BASIS,  
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
//  See the License for the specific language governing permissions and  
//  limitations under the License.  
  
#include <stdio.h>  
#include <stdlib.h>  
#include <string.h>  
#include <math.h>  
#include <pthread.h>  
  
#define MAX_STRING 100  
#define EXP_TABLE_SIZE 1000  
#define MAX_EXP 6  
#define MAX_SENTENCE_LENGTH 1000  
#define MAX_CODE_LENGTH 40  
#define max(a,b) (((a) > (b)) ? (a) : (b))
  
const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary  
const int hownet_hash_size=100000000;
const int wordmeans_hash_size=100000000;
typedef float real;                    // Precision of float numbers  
  
struct vocab_word  
{  
  long long cn;//词频  
  int *point;//huffman编码对应内节点的路径  
  char *word, *code, codelen;//huffman编码  
};  
struct hownet_word
{
  char *word,*means; 
};
struct wordmeans_wordmean{
  long long cn;
  char *word,*mean;
};

char train_file[MAX_STRING], output_file[MAX_STRING];  
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING]; 
char dictionary_file[MAX_STRING];  
struct vocab_word *vocab; 
struct hownet_word *hownet;
struct wordmeans_wordmean *wordmeans;
int binary = 0, cbow = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 1, min_reduce = 1;  
int *vocab_hash; 
int *hownet_hash; 
int *wordmeans_hash; 
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;  
long long hownet_max_size=1000,word_size=0;
long long wordmeans_size=0,wordmeans_max_size=1000;

long long train_words = 0, word_count_actual = 0, file_size = 0, classes = 0;  
real alpha = 0.025, starting_alpha, sample = 0;  
real *syn0, *syn1, *syn1neg, *expTable;  
clock_t start;  
  
int hs = 1, negative = 0;  
const int table_size = 1e8;  
int *table;  
  
  
//每个单词的能量分布表，table在负样本抽样中用到  
//实际上是统计单词的频率
void InitUnigramTable()  
{  
  int a, i;  
  long long train_words_pow = 0;  
  real d1, power = 0.75;  
  table = (int *)malloc(table_size * sizeof(int));  
  for (a = 0; a < vocab_size; a++) //遍历词汇表，统计词的能量总值train_words_pow，指数power应该是缩小值的吧。  
      train_words_pow += pow(vocab[a].cn, power);  
  i = 0;  
  d1 = pow(vocab[i].cn, power) / (real)train_words_pow;//表示已遍历的词的能量值占总能力值的比例  
  for (a = 0; a < table_size; a++)//遍历table。a表示table的位置，i表示词汇表的位置  
  {  
    table[a] = i;//单词i占用table的a位置  
    //table反映的是一个单词能量的分布，一个单词能量越大，所占用的table的位置越多  
    if (a / (real)table_size > d1)  
    {  
      i++;//移动到下一个词  
      d1 += pow(vocab[i].cn, power) / (real)train_words_pow;  
    }  
    if (i >= vocab_size) i = vocab_size - 1;  
  }  
}  
  
// Reads a single word from a file, assuming space + tab + EOL to be word boundaries  
//从文件中读取一个词  
void ReadWord(char *word, FILE *fin) {  
  int a = 0, ch;  
  while (!feof(fin)) {  
    ch = fgetc(fin);  
    if (ch == 13) continue;  //13为回车 
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {  
      if (a > 0) {  
        if (ch == '\n') ungetc(ch, fin);  
        break;  
      }  
      if (ch == '\n') {  
        strcpy(word, (char *)"</s>");  
        return;  
      } else continue;  
    }  
    word[a] = ch;  
    a++;  
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words  
  }  
  word[a] = 0;  
}  

void Readline(char *line,FILE *fin){
  int a = 0, ch;
  while (!feof(fin)) {  
    ch = fgetc(fin);  
    if (ch == 13) continue; 
    if(ch=='\n' ){
      if(a>0)
        break;
      else 
        continue;
    }
    line[a] = ch;  
    a++;  
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words  
  }  
  line[a] = 0; 
}
  
// Returns hash value of a word返回一个词的hash值，一个词跟hash值一一对应（可能冲突）  
int GetWordHash(char *word)  
{  
  unsigned long long a, hash = 0;  
  for (a = 0; a < strlen(word); a++)  
      hash = hash * 257 + word[a];//采取257进制  
  hash = hash % vocab_hash_size;  
  return hash;  
}

int GetWordHash_HowNet(char *word)  
{  
  unsigned long long a, hash = 0;  
  for (a = 0; a < strlen(word); a++)  
      hash = hash * 257 + word[a];//采取257进制  
  hash = hash % hownet_hash_size;  
  return hash;  
}
  

int GetWordMeanHash(char *wordmean)  
{  
  unsigned long long a, hash = 0;  
  for (a = 0; a < strlen(wordmean); a++)  
      hash = hash * 257 + wordmean[a];//采取257进制  
  hash = hash % wordmeans_hash_size;  
  return hash;  
}


int SearchWordMean(char *wordmean){
  unsigned int hash = GetWordMeanHash(wordmean); 
  char wordmean2[2*MAX_STRING];
  while (1)  
  {  
    if (wordmeans_hash[hash] == -1) return -1;  
    strcpy(wordmean2,wordmeans[wordmeans_hash[hash]].word);
    strcat(wordmean2,wordmeans[wordmeans_hash[hash]].mean);
    if (!strcmp(wordmean, wordmean2))  //strcmp如果相等返回值为0
        return wordmeans_hash[hash];  
    hash = (hash + 1) % wordmeans_hash_size;  
  }  
  return -1; 
}
// Returns position of a word in the vocabulary; if the word is not found, returns -1  
// 返回一个词在词汇表中的位置，如果不存在则返回-1  
int SearchVocab(char *word)  
{  
  unsigned int hash = GetWordHash(word);  
  while (1)  
  {  
    if (vocab_hash[hash] == -1) return -1;  
    if (!strcmp(word, vocab[vocab_hash[hash]].word))  //strcmp如果相等返回值为0
        return vocab_hash[hash];  
    hash = (hash + 1) % vocab_hash_size;  
  }  
  return -1;  
}  

int SearchHownet(char *word){
  unsigned int hash = GetWordHash_HowNet(word);  
  while (1)  
  {  
    if (hownet_hash[hash] == -1) return -1;  
    if (!strcmp(word, hownet[hownet_hash[hash]].word))  //strcmp如果相等返回值为0
        return hownet_hash[hash];  
    hash = (hash + 1) % hownet_hash_size;  
  }
  return -1;  
}
  
// Reads a word and returns its index in the vocabulary  
// 从文件流中读取一个词，并返回这个词在词汇表中的位置  
int ReadWordIndex(FILE *fin)  
{  
  char word[MAX_STRING];  
  ReadWord(word, fin);  
  if (feof(fin)) return -1;  
  return SearchVocab(word);  
}  


int AddWordMeanToWordMeans(char *word,char *mean)  
{  
  unsigned int hash, length = max(strlen(word),strlen(mean)) + 1;  
  if (length > MAX_STRING)  
      length = MAX_STRING;  
  //c的内存管理是真的很麻烦
  wordmeans[wordmeans_size].word= (char *)calloc(length, sizeof(char));  
  strcpy(wordmeans[wordmeans_size].word, word);  
  wordmeans[wordmeans_size].mean= (char *)calloc(length, sizeof(char));  
  strcpy(wordmeans[wordmeans_size].mean, mean);
  wordmeans[wordmeans_size].cn = 1;  
  wordmeans_size++;  
  // Reallocate memory if needed  
  if (wordmeans_size + 2 >= wordmeans_max_size)  
  {  
    wordmeans_max_size += 1000;  
    wordmeans= (struct wordmeans_wordmean *)realloc(wordmeans, wordmeans_max_size * sizeof(struct wordmeans_wordmean));  
  }  
  char *wordmean=(char *)calloc(strlen(word)+strlen(mean)+1,sizeof(char));
  strcpy(wordmean,word);
  strcat(wordmean,mean);
  hash =GetWordMeanHash(wordmean);
  while (wordmeans_hash[hash] != -1)//如果hash值冲突了  
      hash = (hash + 1) % wordmeans_hash_size;//使用开放地址法解决冲突  
  wordmeans_hash[hash] = wordmeans_size - 1;//由词的hash值找到她所在词汇表的排序位置  
  return wordmeans_size - 1;  
}


// Adds a word to the vocabulary 将一个词添加到一个词汇中  
int AddWordToVocab(char *word)  
{  
  unsigned int hash, length = strlen(word) + 1;  
  long long hownet_pos;
  if (length > MAX_STRING)  
      length = MAX_STRING;  
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));  
  strcpy(vocab[vocab_size].word, word);  //c的内存管理是真的很麻烦
  vocab[vocab_size].cn = 0;  
  vocab_size++;  
  hownet_pos=SearchHownet(word);
  // Reallocate memory if needed  
  if (vocab_size + 2 >= vocab_max_size)  
  {  
    vocab_max_size += 1000;  
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));  
  }  
  hash = GetWordHash(word);  
  while (vocab_hash[hash] != -1)//如果hash值冲突了  
      hash = (hash + 1) % vocab_hash_size;//使用开放地址法解决冲突  
  vocab_hash[hash] = vocab_size - 1;//由词的hash值找到她所在词汇表的排序位置  
  return vocab_size - 1;  
}  

int AddWordToHowNet(char *word)  
{ 
  unsigned int hash, length = strlen(word) + 1;  
  if (length > MAX_STRING)  
      length = MAX_STRING;  
  hownet[word_size].word = (char *)calloc(length, sizeof(char));  
  strcpy(hownet[word_size].word, word);  //c的内存管理是真的很麻烦 
  word_size++; 
  // Reallocate memory if needed  
  if (word_size + 2 >= hownet_max_size)  
  {  
    hownet_max_size += 1000;  
    hownet = (struct hownet_word *)realloc(hownet, hownet_max_size * sizeof(struct hownet_word)); 
  }  
  hash = GetWordHash_HowNet(word);  
  while (hownet_hash[hash] != -1)//如果hash值冲突了  
      hash = (hash + 1) % hownet_hash_size;//使用开放地址法解决冲突  
  hownet_hash[hash] = word_size - 1;
  //由词的hash值找到她所在词汇表的排序位置  
  return word_size - 1;  
}
  
// Used later for sorting by word counts  
int VocabCompare(const void *a, const void *b)  
{  
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;  
}  

int WordMeanCompare(const void *a,const void *b){ 
  return ((struct wordmeans_wordmean *)b)->cn-((struct wordmeans_wordmean *)a)->cn;
}
  
// Sorts the vocabulary by frequency using word counts  
// 根据词频排序  
void SortVocab()  
{  
  int a, size;  
  unsigned int hash;  
  // Sort the vocabulary and keep </s> at the first position  
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);  
  for (a = 0; a < vocab_hash_size; a++)  //因为vocab已经重新排序了，所以要重新生成HASH表
      vocab_hash[a] = -1;  
  size = vocab_size;  
  train_words = 0;  
  for (a = 0; a < size; a++)  
  {  
    // Words occuring less than min_count times will be discarded from the vocab  
    //出现太少的词直接丢弃  
    if (vocab[a].cn < min_count)  
    {  
      vocab_size--;  
      free(vocab[a].word);
      vocab[a].word=NULL;  
    }  
    else  
    {  
      // Hash will be re-computed, as after the sorting it is not actual  
      // 重新计算hash查找。vocab_hash是由hash值找到该词所在位置  
      hash=GetWordHash(vocab[a].word);  
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;  
      vocab_hash[hash] = a;  
      train_words += vocab[a].cn;  
    }  
  }  
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));  
  // Allocate memory for the binary tree construction  
  for (a = 0; a < vocab_size; a++)  
  {  
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));  
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));  
  }  
}  
 
void SortWordMeans()  
{  

  int a, size;  
  unsigned int hash; 
  char wordmean[2*MAX_STRING];
  // Sort the vocabulary and keep </s> at the first position 
  qsort(&wordmeans[1], wordmeans_size - 1, sizeof(struct wordmeans_wordmean), WordMeanCompare); 
  for (a = 0; a < wordmeans_hash_size; a++)  //因为vocab已经重新排序了，所以要重新生成HASH表
      wordmeans_hash[a] = -1;  
  size = wordmeans_size;  
  for (a = 0; a < size; a++)  
  {  
    // Words occuring less than min_count times will be discarded from the vocab  
    //出现太少的词直接丢弃  
    if (wordmeans[a].cn < min_count)  
    {   

      wordmeans_size--;  
      free(wordmeans[a].word);  
      free(wordmeans[a].mean);
      wordmeans[a].word=NULL;
      wordmeans[a].mean=NULL;

    }  
    else  
    {  
      // Hash will be re-computed, as after the sorting it is not actual  
      // 重新计算hash查找。vocab_hash是由hash值找到该词所在位置 

      strcpy(wordmean,wordmeans[a].word);
      strcat(wordmean,wordmeans[a].mean);
      hash=GetWordMeanHash(wordmean);  
      while (wordmeans_hash[hash] != -1) hash = (hash + 1) % wordmeans_hash_size;  
      wordmeans_hash[hash] = a;  
     //train_words += wordmeans[a].cn;  
    }  
  }
  wordmeans= (struct wordmeans_wordmean *)realloc(wordmeans, (wordmeans_size + 1) * sizeof(struct wordmeans_wordmean));  
  // Allocate memory for the binary tree construction   
}

void ReduceWordMeans()  
{  
  int a, b = 0;  
  unsigned int hash; 
  unsigned int word_hash;  
  char *word;
  int pos;
  for (a = 0; a < wordmeans_size; a++){//我草，这很容易看错啊   //的确容易看错
    if (wordmeans[a].cn > min_reduce)  
    {  
      wordmeans[b].cn = wordmeans[a].cn;  
      wordmeans[b].word = wordmeans[a].word;
      wordmeans[b].mean = wordmeans[a].mean;    
      b++;  
    } 
    else {
       free(wordmeans[a].word);
       free(wordmeans[a].mean);
    }
  }
  wordmeans_size = b;  
  for (a = 0; a < wordmeans_hash_size; a++) wordmeans_hash[a] = -1;  //再次重新生成hash
  for (a = 0; a < wordmeans_size; a++) {  
    // Hash will be re-computed, as it is not actual 
    char wordmean[2*MAX_STRING ];
    strcpy(wordmean,wordmeans[wordmeans_hash[hash]].word);
    strcat(wordmean,wordmeans[wordmeans_hash[hash]].mean);
    hash = GetWordMeanHash(wordmean);  
    while (wordmeans_hash[hash] != -1) hash = (hash + 1) % wordmeans_hash_size;  
    wordmeans_hash[hash] = a;  
  }  
  fflush(stdout);  //ff(stdin)刷新标准输入缓冲区，把输入缓冲区里的东西丢弃   
           //fflush(stdout)刷新标准输出缓冲区，把输出缓冲区里的东西打印到标准输出设备上。
  //min_reduce++;  
  
} 
// Reduces the vocabulary by removing infrequent tokens  
// 再次移除词频过小的词，缩减词汇表  
void ReduceVocab()  
{  
  int a, b = 0;  
  unsigned int hash;  
  for (a = 0; a < vocab_size; a++)//我草，这很容易看错啊   //的确容易看错
  if (vocab[a].cn > min_reduce)  
  {  
    vocab[b].cn = vocab[a].cn;  
    vocab[b].word = vocab[a].word;  
    b++;  
  }  
  else free(vocab[a].word);  
  vocab_size = b;  
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;  //再次重新生成hash
  for (a = 0; a < vocab_size; a++) {  
    // Hash will be re-computed, as it is not actual  
    hash = GetWordHash(vocab[a].word);  
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;  
    vocab_hash[hash] = a;  
  }  
  fflush(stdout);  //ff(stdin)刷新标准输入缓冲区，把输入缓冲区里的东西丢弃   
           //fflush(stdout)刷新标准输出缓冲区，把输出缓冲区里的东西打印到标准输出设备上。
  //min_reduce++;  
}  
  

void ImportHowNetMeans( char *file){
  unsigned int length;
  int count,pos;
  int i,a;
  for (a = 0; a < hownet_hash_size; a++) hownet_hash[a] = -1; 
  FILE *fin=fopen(file,"rb");
  char means[2*MAX_STRING];
  char word[MAX_STRING];
  while(!feof(fin)){
      ReadWord(word,fin);
      Readline(means,fin);
      i = SearchHownet(word);//返回该词在词汇表中的位置  
     if (i == -1)//该词之前不存在  
      {  
        a = AddWordToHowNet(word);
        length = strlen(means) + 1;  
        if (length > 5*MAX_STRING)  
          length = 5*MAX_STRING; 
         hownet[a].means = (char *)calloc(length, sizeof(char));
         strcpy(hownet[a].means,means);
     //把该词添加到词汇表中  //use the array to get meaning
      }  
  }
  fclose(fin);
}
// Create binary Huffman tree using the word counts根据词频创建huffman树  
// Frequent words will have short uniqe binary codes词频越大的单词有越短的huffman编码  
void CreateBinaryTree() {  
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];  
  char code[MAX_CODE_LENGTH];  
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));  
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));  
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));  
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;  
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;  //非叶子节点
  pos1 = vocab_size - 1;  
  pos2 = vocab_size;  
  // Following algorithm constructs the Huffman tree by adding one node at a time  
  for (a = 0; a < vocab_size - 1; a++)  
  {  
    // First, find two smallest nodes 'min1, min2' 找出目前权值最小的两个节点  
    if (pos1 >= 0)//第一个权值最小的节点  //这个查找很有意思，通过俩个临时变量 //有个屁意思，已经sort好了
    {  
      if (count[pos1] < count[pos2])  
      {  
        min1i = pos1;  
        pos1--;  
      }  
      else  
      {  
        min1i = pos2;  
        pos2++;  
      }  
    }  
    else  
    {  
      min1i = pos2;  
      pos2++;  
    }  
    if (pos1 >= 0)//第二个权值最小的节点  
    {  
      if (count[pos1] < count[pos2])  
      {  
        min2i = pos1;  
        pos1--;  
      }  
      else  
      {  
        min2i = pos2;  
        pos2++;  
      }  
    }  
    else  
    {  
      min2i = pos2;  
      pos2++;  
    }  
    count[vocab_size + a] = count[min1i] + count[min2i];  
    parent_node[min1i] = vocab_size + a;  
    parent_node[min2i] = vocab_size + a;  
    binary[min2i] = 1;//节点编码为1，之前默认是0。  //机智
  }  
  // Now assign binary code to each vocabulary word  
 //分配编码，非常有用
  for (a = 0; a < vocab_size; a++)  
  {  
    b = a;  
    i = 0;  
    while (1)  
    {  
      code[i] = binary[b];  
      point[i] = b;  
      i++;  
      b = parent_node[b];  
      if (b == vocab_size * 2 - 2) break;  
    }  
    vocab[a].codelen = i;  
    vocab[a].point[0] = vocab_size - 2;  
    for (b = 0; b < i; b++)//reverse the order  
    {  
      vocab[a].code[i - b - 1] = code[b];  
      vocab[a].point[i - b] = point[b] - vocab_size;  
    }  
  }  
  free(count);  
  free(binary);  
  free(parent_node);  
}  
  
//从分词文件中统计每个单词的词频  
void LearnVocabFromTrainFile()  
{  
  char word[MAX_STRING];  
  FILE *fin;  
  long long a, i; 
  long long hownet_pos;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1; 
  for(a=0;a<wordmeans_hash_size;a++)wordmeans_hash[a]=-1; 
  char means[2*MAX_STRING];
  char *mean;
  char wordmean[2*MAX_STRING];
  int pos;
  fin = fopen(train_file, "rb");  
  if (fin == NULL)  
  {  
    printf("ERROR: training data file not found!\n");  
    exit(1);  
  }  
  vocab_size = 0;
  wordmeans_size = 0;   
  AddWordToVocab((char *)"</s>");  
  while (1)  
  {  
    ReadWord(word, fin); 
    if (feof(fin)) break;  
    train_words++;  
    if ((debug_mode > 1) && (train_words % 100000 == 0))  
    {  
      printf("%lldK%c", train_words / 1000, 13);  
      fflush(stdout);  
    }
    hownet_pos=SearchHownet(word);
    i = SearchVocab(word);//返回该词在词汇表中的位置  
    if (hownet_pos== -1)//can't find the word  
    {  
      continue;
    }  
    else{
      if(i==-1){
        a=AddWordToVocab(word);
        vocab[a].cn=1;
      }
      else{
        vocab[i].cn++;
      }//get the meanings
      const char *split=" ";
      strcpy(means,hownet[hownet_pos].means);
      strcpy(word,hownet[hownet_pos].word);
      mean=strtok(means,split);
      strcpy(wordmean,word);
      strcat(wordmean,mean);
      pos=SearchWordMean(wordmean);
      if(pos==-1){
        a = AddWordMeanToWordMeans(word,mean);  
        wordmeans[a].cn = 1;
      }
      else{
        wordmeans[pos].cn++;
      }
      mean = strtok(NULL,split);
      while(mean!=NULL && strcmp(mean," ")) { 
        strcpy(wordmean,word);
        strcat(wordmean,mean);
        pos=SearchWordMean(wordmean);
        if(pos==-1){
          a = AddWordMeanToWordMeans(word,mean);  
  
          wordmeans[a].cn = 1;
        }
        else{   
          wordmeans[pos].cn++;
        }
        mean = strtok(NULL,split);
      }
      if (vocab_size > vocab_hash_size * 0.7){//如果词汇表太庞大，就缩减 //because the wordmeans and vocab have similary process of increasement    
          ReduceWordMeans();
          ReduceVocab();
      } 
    }
  }

  printf("%lld,%lld\n",train_words,vocab_size);
   printf("%lld\n",wordmeans_size);
  SortVocab();
  SortWordMeans(); 
  file_size = ftell(fin);  
  fclose(fin);  
}  
  
void SaveVocab() {  
  long long i;  
  FILE *fo = fopen(save_vocab_file, "wb");  
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);  
  fclose(fo);  
}  
  
//从文件读取词汇，该文件已经统计好了每个词汇的词频  
void ReadVocab()  
{  
  long long a, i = 0;  
  char c;  
  char word[MAX_STRING];  
  FILE *fin = fopen(read_vocab_file, "rb");//打开词汇文件  
  if (fin == NULL)  
  {  
    printf("Vocabulary file not found\n");  
    exit(1);  
  }  
  for (a = 0; a < vocab_hash_size; a++)  
      vocab_hash[a] = -1;  
  vocab_size = 0;  
  while (1)  
  {  
    ReadWord(word, fin);//从fin进入一个词到word中  
    if (feof(fin)) break;  
    a = AddWordToVocab(word);//把该词添加到词汇中，并返回该词的位置  
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);//读取词频？c是干啥的吗，读取空格吗  //应该是，总不能不读取吧
    i++;  
  }  
  SortVocab();//根据词频排序  
  if (debug_mode > 0)  
  {  
    printf("Vocab size: %lld\n", vocab_size);  
    printf("Words in train file: %lld\n", train_words);  
  }  
  
  //读取训练数据  
  fin = fopen(train_file, "rb");  
  if (fin == NULL)  
  {  
    printf("ERROR: training data file not found!\n");  
    exit(1);  
  }  
  fseek(fin, 0, SEEK_END);  
  file_size = ftell(fin);  
  fclose(fin);  
}  
  
void InitNet()  
{  
  long long a, b;  
  a = posix_memalign((void **)&syn0, 128, (long long)wordmeans_size * layer1_size * sizeof(real));  
  //先知道这个也是申请动态数组，对齐还有128这个参数以后再了解  
  /*int posix_memalign (void **memptr,
                    size_t alignment,
                    size_t size);
  * See http://perens.com/FreeSoftware/ElectricFence/ and http://valgrind.org, respectively.
  调用posix_memalign( )成功时会返回size字节的动态内存，并且这块内存的地址是alignment的倍数。参数alignment必须是2的幂，还是void指针的大小的  倍数。返回的内存块的地址放在了memptr里面，函数返回值是0.*/
  if (syn0 == NULL)  
  {  
      printf("Memory allocation failed\n"); exit(1);  
  }  
  if (hs)//采用hierachical softmax  
  {  
    a = posix_memalign((void **)&syn1, 128, (long long)wordmeans_size * layer1_size * sizeof(real));  
    if (syn1 == NULL)  
    {  
        printf("Memory allocation failed\n"); exit(1);  
    }  
  //没有理解这是要干什么？
  //layer1_size是什么？
//layer1_size词向量的维度，默认值是100  
    for (b = 0; b < layer1_size; b++)  
        for (a = 0; a < wordmeans_size; a++)  
            syn1[a * layer1_size + b] = 0;  
  }  
  if (negative>0)//还有负样本  
  {  
    a = posix_memalign((void **)&syn1neg, 128, (long long)wordmeans_size * layer1_size * sizeof(real));  
    if (syn1neg == NULL)  
    {  
        printf("Memory allocation failed\n"); exit(1);  
    }  
    for (b = 0; b < layer1_size; b++)  
        for (a = 0; a < wordmeans_size; a++)  
            syn1neg[a * layer1_size + b] = 0;  
  }  
//初始化词向量，一共有vocab_size个单词，每个单词词向量维度是layer1_size
  for (b = 0; b < layer1_size; b++)  
      for (a = 0; a < wordmeans_size; a++)  
          syn0[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;  //随机初始化词向量
  CreateBinaryTree();//建立huffman树，对每个单词进行编码  
}  
  
void DestroyNet() {
  if (syn0 != NULL) {
    free(syn0);
  }
  if (syn1 != NULL) {
    free(syn1);
  }
  if (syn1neg != NULL) {
    free(syn1neg);
  }
}

  
//这个线程函数执行之前，已经做好了一些工作：根据词频排序的词汇表，每个单词的huffman编码  
void *TrainModelThread(void *id)  
{  
  long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;  
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];  
  long long l1, l2, c, target, label;  
  unsigned long long next_random = (long long)id;
  int hownet_pos;  
  char means[2*MAX_STRING];
  char wordstr[MAX_STRING];
  char mean[MAX_STRING];
  char wordmean[2*MAX_STRING];
  int des_pos;
  int res_pos;
  long long wordmean_pos;
  real f, g;  
  clock_t now;  
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));  //这是什么？
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));  
  FILE *fi = fopen(train_file, "rb");  
  //每个线程对应一段文本。根据线程id找到自己负责的文本的初始位置  
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);  //每个线程处理一段文本
  while (1)  
  {  
    if (word_count - last_word_count > 10000)  //每算到10000个单词记录以下
    {  
      word_count_actual += word_count - last_word_count;  
      last_word_count = word_count;  
      if ((debug_mode > 1))  
      {  
        now=clock();  
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,  
         word_count_actual / (real)(train_words + 1) * 100,  
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));  
        fflush(stdout);  
      }  
     // alpha是随着已经学习的单词数量递减,每经过10000单词更新一次
      alpha = starting_alpha * (1 - word_count_actual / (real)(train_words + 1));  
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;  //alpha的最低限制
    }  
    if (sentence_length == 0)  
    {  
      while (1)  
      {  
        word = ReadWordIndex(fi);//从文件流中读取一个词，并返回这个词在词汇表中的位置  
        if (feof(fi)) break;  
        if (word == -1) continue;  
        word_count++;  
        if (word == 0) break;  
        // The subsampling randomly discards frequent words while keeping the ranking same  
        if (sample > 0)//对高频词进行下采样to ignore some instance of a word，不过要保持排序不变。  
        {  
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;  
          next_random = next_random * (unsigned long long)25214903917 + 11;  
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;  
        }  
        sen[sentence_length] = word;  
        sentence_length++;  
        //1000个单词视作一个句子？  //我也觉得是
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;  
      }  
      sentence_position = 0;  
    }  
    if (feof(fi)) break;  
    if (word_count > train_words / num_threads) break;//如果当前线程已处理的单词超过了 阈值，则退出。  
    word = sen[sentence_position];  
    if (word == -1) continue;  
   //这个是啥？！！！
   //neu1 stores the sum the input word verctors as the input vector
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;  //应该是中间的特征向量什么的？暂时不确定
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;  
    next_random = next_random * (unsigned long long)25214903917 + 11;  
  //基于伪随机数的算法 http://www.cnblogs.com/xkfz007/archive/2012/08/25/2656893.html
    b = next_random % window;  
    if (cbow)  
    {  //train the cbow architecture  
      // in -> hidden  
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window)//扫描目标单词的左右几个单词  //为什么要随机扫描？即为什么b是随机生成的？
      {  
        c = sentence_position - window + a;  
        if (c < 0) continue;  
        if (c >= sentence_length) continue;  
        last_word = sen[c];  
        if (last_word == -1) continue;  
        hownet_pos=SearchHownet(vocab[last_word].word);
        if(hownet_pos==-1)
          continue;
        strcpy(means,hownet[hownet_pos].means);
        res_pos=0;
        des_pos=0;
        while(1)
        { 
          if(means[res_pos]=='\0' ) break;
          if(means[res_pos]==' ' ) 
          {
            mean[des_pos]='\0';
            strcpy(wordmean,vocab[word].word);
            strcat(wordmean,mean);            
            wordmean_pos=SearchWordMean(wordmean);
            c=wordmean_pos;
            for (c = 0; c < layer1_size; c++)//layer1_size词向量的维度，默认值是100  
              neu1[c] += syn0[c + last_word * layer1_size];
            des_pos=0;
            res_pos++;

          }
          else
            mean[des_pos++]=means[res_pos++];
       }
       //两个循环一起计算词向量的和
        
      }  
      if (hs) for (d = 0; d < vocab[word].codelen; d++)//开始遍历huffman树，每次一个节点  
      {  
        f = 0;  
        l2 = vocab[word].point[d] * layer1_size;//point应该记录的是huffman的路径。找到当前节点，并算出偏移  每一个词向量是占据一个layer1_size的
        // Propagate hidden -> output  
        for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];//计算内积  
        if (f <= -MAX_EXP) continue;//内积不在范围内直接丢弃  
        else if (f >= MAX_EXP) continue;  
        else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];//内积之后sigmoid函数  //expTable 是提前计算好的数值表
        // 'g' is the gradient multiplied by the learning rate  
        g = (1 - vocab[word].code[d] - f) * alpha;//偏导数的一部分  
  
        //layer1_size是向量的维度  
        // Propagate errors output -> hidden 反向传播误差，从huffman树传到隐藏层。下面就是把当前内节点的误差（累加才是重点）传播给隐藏层，syn1[c + l2]是偏导数的一部分。  
        for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];  
  
        // Learn weights hidden -> output 更新当前内节点的向量，后面的neu1[c]其实是偏导数的一部分  
        for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];  
      }  
      // NEGATIVE SAMPLING  
      if (negative > 0)  
      for (d = 0; d < negative + 1; d++)  
      {  
        if (d == 0)  
        {  
          target = word;//目标单词  
          label = 1;//正样本  
        }  
        else  
        {  
          next_random = next_random * (unsigned long long)25214903917 + 11;  
          target = table[(next_random >> 16) % table_size];  
          if (target == 0) target = next_random % (vocab_size - 1) + 1;  
          if (target == word) continue;  
          label = 0;//负样本  
        }  
        l2 = target * layer1_size;  
        f = 0;  
        for (c = 0; c < layer1_size; c++)  
            f += neu1[c] * syn1neg[c + l2];//内积  
        if (f > MAX_EXP)  
            g = (label - 1) * alpha;  
        else if (f < -MAX_EXP)  
            g = (label - 0) * alpha;  
        else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;  
        for (c = 0; c < layer1_size; c++)  
            neu1e[c] += g * syn1neg[c + l2];//隐藏层的误差  
        for (c = 0; c < layer1_size; c++)  
            syn1neg[c + l2] += g * neu1[c];//更新负样本向量  
      }  
      // hidden -> in  
      for (a = b; a < window * 2 + 1 - b; a++)  
      if (a != window)//cbow模型 更新的不是中间词语的向量，而是周围几个词语的向量。  
      {  
        c = sentence_position - window + a;  
        if (c < 0) continue;  
        if (c >= sentence_length) continue;  
        last_word = sen[c];  
        if (last_word == -1) continue;  
        for (c = 0; c < layer1_size; c++)  
            syn0[c + last_word * layer1_size] += neu1e[c];//更新词向量  
      }  
    }  
    else  
    {  //train skip-gram  
       for (a = b; a < window * 2 + 1 - b; a++)  
       if (a != window)//扫描周围几个词语  
       {  
        c = sentence_position - window + a;  
        if (c < 0) continue;  
        if (c >= sentence_length) continue;  
        last_word = sen[c];  
        if (last_word == -1) continue;  
        l1 = last_word * layer1_size;  
        for (c = 0; c < layer1_size; c++)  
            neu1e[c] = 0;  
        // HIERARCHICAL SOFTMAX  
        if (hs)  
        for (d = 0; d < vocab[word].codelen; d++)//遍历叶子节点  
        {  
          f = 0;  
          l2 = vocab[word].point[d] * layer1_size;//point记录的是huffman的路径  
          // Propagate hidden -> output 感觉源代码这个英语注释有点误导人，这里的隐藏层就是输入层，就是词向量。  
          for (c = 0; c < layer1_size; c++)  
              f += syn0[c + l1] * syn1[c + l2];//计算两个词向量的内积  
          if (f <= -MAX_EXP) continue;  
          else if (f >= MAX_EXP) continue;  
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];  
          // 'g' is the gradient multiplied by the learning rate  
          g = (1 - vocab[word].code[d] - f) * alpha;//偏导数的一部分  
          // Propagate errors output -> hidden  
          for (c = 0; c < layer1_size; c++)  
              neu1e[c] += g * syn1[c + l2];//隐藏层的误差  
          // Learn weights hidden -> output  
          for (c = 0; c < layer1_size; c++)  
              syn1[c + l2] += g * syn0[c + l1];//更新叶子节点向量  
        }  
        // NEGATIVE SAMPLING  
        if (negative > 0)//这个同cobow差不多  
        for (d = 0; d < negative + 1; d++)  
        {  
          if (d == 0)  
          {  
            target = word;  
            label = 1;  
          }  
          else  
          {  
            next_random = next_random * (unsigned long long)25214903917 + 11;  
            target = table[(next_random >> 16) % table_size];  
            if (target == 0) target = next_random % (vocab_size - 1) + 1;  
            if (target == word) continue;  
            label = 0;  
          }  
          l2 = target * layer1_size;  
          f = 0;  
          for (c = 0; c < layer1_size; c++)  
              f += syn0[c + l1] * syn1neg[c + l2];  
          if (f > MAX_EXP) g = (label - 1) * alpha;  
          else if (f < -MAX_EXP)  
              g = (label - 0) * alpha;  
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;  
          for (c = 0; c < layer1_size; c++)  
              neu1e[c] += g * syn1neg[c + l2];  
          for (c = 0; c < layer1_size; c++)  
              syn1neg[c + l2] += g * syn0[c + l1];  
        }  
  
        // Learn weights input -> hidden  
        for (c = 0; c < layer1_size; c++)  
            syn0[c + l1] += neu1e[c];//更新周围几个词语的向量  
      }  
    }  
    sentence_position++;  
    if (sentence_position >= sentence_length)  
    {  
      sentence_length = 0;  
      continue;  
    }  
  }  
  fclose(fi);  
  free(neu1);  
  free(neu1e);  
  pthread_exit(NULL);  
}  
  
void TrainModel()  
{  
  long a, b, c, d;  
  FILE *fo;  
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));  
  printf("Starting training using file %s\n", train_file);  
  starting_alpha = alpha;  
  if (read_vocab_file[0] != 0)  
      ReadVocab();//从文件读入词汇  
  else  
      LearnVocabFromTrainFile();//从训练文件学习词汇  
  if (save_vocab_file[0] != 0)  
      SaveVocab();//保存词汇  
  if (output_file[0] == 0)  
      return;  
  InitNet();  
  if (negative > 0) InitUnigramTable();  
  start = clock();  
  //函数pthread_create用来创建一个线程，它的原型为：
  //extern int pthread_create __P ((pthread_t *__thread, __const pthread_attr_t *__attr,

  //　void *(*__start_routine) (void *)， void *__arg));
  //第一个参数为指向线程标识符的指针，第二个参数用来设置线程属性，第三个参数是线程运行函数的起始地址，最后一个参数是运行函数的参数。
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);  
//下面一个问题和前面创建进程类似，不过带来的问题会比进程要严重得多。如果你的主线程，也就是main函数执行的那个线程，在你其他县城推出之前就已经退出，那么带来的bug则不可估量。通过pthread_join函数会让主线程阻塞，直到所有线程都已经退出。
 
//int pthread_join(pthread_t thread, void **value_ptr);
 
//thread：等待退出线程的线程号。
 
//value_ptr：退出线程的返回值。
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);  
  fo = fopen(output_file, "wb");  
  if (classes == 0) //不需要聚类，只需要输出词向量  
  {  
    // Save the word vectors  
    fprintf(fo, "%lld %lld\n", wordmeans_size, layer1_size);  
    for (a = 0; a < wordmeans_size; a++)  
    {  
      fprintf(fo, "%s ", wordmeans[a].word); 
      fprintf(fo, "%s ", wordmeans[a].mean);   
      if (binary)  
          for (b = 0; b < layer1_size; b++)  
              fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);  
      else  
          for (b = 0; b < layer1_size; b++)  
              fprintf(fo, "%lf ", syn0[a * layer1_size + b]);  
      fprintf(fo, "\n");  
    }  
  }  
  else //使用k-means进行聚类  
  {  
    // Run K-means on the word vectors  
    int clcn = classes, iter = 10, closeid;  
    int *centcn = (int *)malloc(classes * sizeof(int));//该类别的数量  
    int *cl = (int *)calloc(vocab_size, sizeof(int));//词到类别的映射  
    real closev, x;  
    real *cent = (real *)calloc(classes * layer1_size, sizeof(real));//质心数组  
    for (a = 0; a < vocab_size; a++)  
        cl[a] = a % clcn;//任意分类？  
    for (a = 0; a < iter; a++)  
    {  
      for (b = 0; b < clcn * layer1_size; b++)  
          cent[b] = 0;//质心清零  
      for (b = 0; b < clcn; b++)  
          centcn[b] = 1;  
      for (c = 0; c < wordmeans_size; c++)  
      {  
        for (d = 0; d < layer1_size; d++)  
            cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];//求和放到质心数组中  
        centcn[cl[c]]++;//类别数量加1  
      }  
      for (b = 0; b < clcn; b++)//遍历所有类别  
      {  
        closev = 0;  
        for (c = 0; c < layer1_size; c++)  
        {  
          cent[layer1_size * b + c] /= centcn[b];//均值，就是求新的质心  
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];  
        }  
        closev = sqrt(closev);  
        for (c = 0; c < layer1_size; c++)  
            cent[layer1_size * b + c] /= closev;//对质心进行归一化？  
      }  
      for (c = 0; c < wordmeans_size; c++)//对所有词语重新分类  
      {  
        closev = -10;  
        closeid = 0;  
        for (d = 0; d < clcn; d++)  
        {  
          x = 0;  
          for (b = 0; b < layer1_size; b++)  
              x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];//内积  
          if (x > closev)  
          {  
            closev = x;  
            closeid = d;  
          }  
        }  
        cl[c] = closeid;  
      }  
    }  
    // Save the K-means classes  
    for (a = 0; a < wordmeans_size; a++)  
        fprintf(fo, "%s  %s %d\n", wordmeans[a].word,wordmeans[a].mean, cl[a]);  
    free(centcn);  
    free(cent);  
    free(cl);  
  }  
  fclose(fo);  
}  
  
int ArgPos(char *str, int argc, char **argv)  
{  
  int a;  
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a]))  
  {  
    if (a == argc - 1)  
    {  
      printf("Argument missing for %s\n", str);  
      exit(1);  
    }  
    return a;  
  }  
  return -1;  
}  
  
int main(int argc, char **argv) {  
  int i;  
  if (argc == 1) {  
    printf("WORD VECTOR estimation toolkit v 0.1b\n\n");  
    printf("Options:\n");  
    printf("Parameters for training:\n");  
  

    printf("\t -dict <dictionary file> \n");
    printf("\t\t Use word and it's meanings from <dictionary file> to train the model");
    //输入文件：已分词的语料  
    printf("\t-train <file>\n");  
    printf("\t\tUse text data from <file> to train the model\n");  
  
    //输出文件：词向量或者词聚类  
    printf("\t-output <file>\n");  
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");  
  
    //词向量的维度，默认值是100  
    printf("\t-size <int>\n");  
    printf("\t\tSet size of word vectors; default is 100\n");  
  
    //窗口大小，默认是5  
    printf("\t-window <int>\n");  
    printf("\t\tSet max skip length between words; default is 5\n");  
  
    //设定词出现频率的阈值，对于常出现的词会被随机下采样  
    printf("\t-sample <float>\n");  
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency");  
    printf(" in the training data will be randomly down-sampled; default is 0 (off), useful value is 1e-5\n");  
  
    //是否采用softmax体系  
    printf("\t-hs <int>\n");  
    printf("\t\tUse Hierarchical Softmax; default is 1 (0 = not used)\n");  
  
    //负样本的数量，默认是0，通常使用5-10。0表示不使用。  
    printf("\t-negative <int>\n");  
    printf("\t\tNumber of negative examples; default is 0, common values are 5 - 10 (0 = not used)\n");  
  
    //开启的线程数量  
    printf("\t-threads <int>\n");  
    printf("\t\tUse <int> threads (default 1)\n");  
  
    //最小阈值。对于出现次数少于该值的词，会被抛弃掉。  
    printf("\t-min-count <int>\n");  
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");  
  
    //学习速率初始值，默认是0.025  
    printf("\t-alpha <float>\n");  
    printf("\t\tSet the starting learning rate; default is 0.025\n");  
  
    //输出词类别，而不是词向量  
    printf("\t-classes <int>\n");  
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");  
  
    //debug模式，默认是2，表示在训练过程中会输出更多信息  
    printf("\t-debug <int>\n");  
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");  
  
    //是否用binary模式保存数据，默认是0，表示否。  
    printf("\t-binary <int>\n");  
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");  
  
    //保存词汇到这个文件  
    printf("\t-save-vocab <file>\n");  
    printf("\t\tThe vocabulary will be saved to <file>\n");  
  
    //词汇从该文件读取，而不是由训练数据重组  
    printf("\t-read-vocab <file>\n");  

    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");  
  
    //是否采用continuous bag of words算法。默认是0，表示采用另一个叫skip-gram的算法。  
    printf("\t-cbow <int>\n");  
    printf("\t\tUse the continuous bag of words model; default is 0 (skip-gram model)\n");  
  
    //工具使用样例  
    printf("\nExamples:\n");  
    printf("./word2vec -train data.txt -output vec.txt -debug 2 -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1\n\n");  
    return 0;  
  }  
  output_file[0] = 0;  
  save_vocab_file[0] = 0;  
  read_vocab_file[0] = 0;  
  if ((i =ArgPos((char *)"-dict",argc,argv))>0) strcpy(dictionary_file,argv[i+1]);
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);  
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);  
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);  
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);  
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);  
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);  
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);  
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);  
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);  
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);  
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);  
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);  
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);  
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);  
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);  
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);  
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  wordmeans_hash = (int *)calloc(wordmeans_hash_size, sizeof(int));  
  hownet_hash=(int *)calloc(hownet_hash_size,sizeof(int));
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  hownet=(struct hownet_word*)calloc(hownet_max_size,sizeof(struct hownet_word));
  wordmeans=(struct wordmeans_wordmean*)calloc(wordmeans_max_size,sizeof(struct wordmeans_wordmean));  
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real)); 
  for (i = 0; i < EXP_TABLE_SIZE; i++)  
  {  
    //expTable[i] = exp((i -500)/ 500 * 6) 即 e^-6 ~ e^6  
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table  
    //expTable[i] = 1/(1+e^6) ~ 1/(1+e^-6)即 0.01 ~ 1 的样子  
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)  
  }  
  ImportHowNetMeans(dictionary_file);
  TrainModel();  
  DestroyNet();
  free(vocab_hash);
  free(expTable);
  return 0;  
}  
