from ray.core.generated.gcs_pb2 import OBJECT
import ray
from requests import patch
from tqdm import tqdm

from src.dataset.loader import Loader

TEST_SEGMENTS = ['atgtaaggcc', 'aaggcctt', 'cacatgaacc', 'ctgttgcaaa', 'ggcctttgaa', 'attcaaaggc', 'ttcaaaggcc',
                 'catgaaccca', 'tgtaaggcct', 'agttagcgat', 'cactgttgca', 'tgcaacagtg', 'gtaagcagag', 'gtgggccgta',
                 'accgactatt', 'aaggccggca', 'gcctttgaat', 'agaaaaaa', 'actgttgcaa', 'ggcactgttg', 'tcaaaaactc',
                 'cctttgaata', 'cttattcaaa', 'aggccttaca', 'ccttacattt', 'gggttcatgt', 'cagtactg', 'cccgcggg',
                 'aaaggcctta', 'aaggccttac', 'acagggccta', 'tacatttcaa', 'ggtcctcgtt', 'aatgtaaggc', 'agaaaaaac',
                 'tcccgcggga', 'tatagctata', 'ttgaaatgta', 'aaggcctttg', 'ccgactattt', 'tagtgtttt', 'aaatattt',
                 'tggagctgca', 'gaggacctat', 'atacgtat', 'gcaattgc', 'gccatatggc', 'ttgtggaggg', 'gcgtcgcatc',
                 'gaaaaaaa', 'aacgcaat', 'ctccaccgat', 'taatagacac', 'gtacaggata', 'gatcatgatc', 'tttgcaacag',
                 'tgaggcagcc', 'gcacagaggc', 'gccaacggta', 'gccaaggc', 'tacggcccac', 'catgcatg', 'aaatatcct',
                 'gtagtggtac', 'tcacgtga', 'taacgtta', 'gcgagagtcg', 'taatggct', 'gcctgcaggc', 'ttgatcaa', 'gtcggtgc',
                 'ttgagaaaac', 'tcctgcagga', 'ccgttacgca', 'gagctactgc', 'ggtcccacga', 'gccgcggc', 'aatgcccgat',
                 'ataggtcctc', 'tctttagac', 'cttggccatc', 'gcactacag', 'agtatact', 'gtatcttcga', 'agtacttctt',
                 'cgaaccagga', 'agcactcagg', 'tcgggacggc', 'gaaaattttc', 'ctcgtacgag', 'acgagctcgt', 'acctcaaaaa',
                 'tttcagactg', 'tttttaaaaa', 'tcatacgctc', 'cctctaaact', 'gcatctcata', 'ggggcccc', 'cccaaagagc',
                 'tgatatca', 'tgcaacagt', 'ttcatgaa', 'cccctagg', 'tgcagctcca', 'aagaggaggt', 'atgcgcat', 'tgtgcagctc',
                 'tgatatatca', 'gatcgtacga', 'ttccaccgt', 'actgcttctg', 'tgattttttt', 'gtaatattac', 'atttcaaaag',
                 'gcttaagc', 'ctttaaag', 'gaatcgattc', 'gattttgag', 'tgctacgtcc', 'agatcgatct', 'atattggtca',
                 'ctgcgcag', 'gttgctca', 'taaggccttt', 'gaggcctca', 'ctgttacttg', 'ctttgcaaca', 'ccatcataca',
                 'actcaattc', 'aaatattcta', 'cctcaaaaa', 'tgtgggccgt', 'cacgcactga', 'aggcagcctt', 'ttctttcttt',
                 'taggccta', 'cgctgacccg', 'caacttat', 'gttttgga', 'catgatataa', 'cagacagtca', 'ccagtactgg',
                 'aggccggcat', 'atgcctgttg', 'gtcccacga', 'gctcaagcga', 'aagccactg', 'tcgcatgcga', 'ccagaacaag',
                 'tccgtccca', 'tccttaagga', 'cagcatgctg', 'gaagatagg', 'atcagctgtc', 'ggtcatgacc', 'ctttatgttg',
                 'gctttatgag', 'cggtacctgc', 'accgaagtc', 'gttggataac', 'cttctttaga', 'acgcggtcga', 'cattattgtt',
                 'acggtacgct', 'caaaagacca', 'actgttgca', 'taccatggta', 'caaaaactct', 'ggatgcggcg', 'ccatggtgct',
                 'gctcttgaga', 'tagtgtttgc', 'ccccaaggcg', 'tttccggaaa', 'agtaagttgg', 'aggttcgttg', 'cagtagctg',
                 'aagattacat', 'gggcggtggg', 'cgtacggccc', 'gttcgaac', 'tggcagggct', 'ttcaaaaact', 'cctgtacagg',
                 'caggatcgta', 'ggtcactttg', 'catttggaag', 'aaggtggtcg', 'acggcaattc', 'ttacgcacca', 'ccattcaaag',
                 'gtcatgac', 'gggcggct', 'tcagcggag', 'cgaccccc', 'ctaatggctc', 'gcgatcgc', 'tcaaataatc', 'tagatcta',
                 'aaattttaag', 'tcttaacgag', 'gcccctagg', 'cagctccac', 'agtaaacgtt', 'actgaagagg', 'aatgctaggc',
                 'agcgagctgg', 'agtcctaccc', 'tcaattga', 'cggccaag', 'tatctcctc', 'tccgtgactg', 'gtaagttggc',
                 'agctttgat', 'gagctcggat', 'aggcgatgtt', 'ttattcaaag', 'ttgcaagatc', 'cgaaatttcg', 'tgactccaa',
                 'actatgcacc', 'cccggccggg', 'gtattagaag', 'tatgcata', 'tactgaaca', 'gaaaacat', 'ttcaagtggc',
                 'tttgagaa', 'tccgtcccag', 'gaacttgcaa', 'ctcggacaca', 'atccttggat', 'atgcttggaa', 'cgtatacg',
                 'cacttggat', 'cagagttttt', 'ccaccgat', 'taaggcctt', 'ttagccccg', 'ggtgcgcacc', 'aggctcga', 'agtatccta',
                 'gcttaccagg', 'cggctgcag', 'tacattgcaa', 'cccaaggcgt', 'gctgcctcat', 'acttgcaaat', 'ccagattc',
                 'cgtgacaac', 'gattatagcg', 'aaccccatg', 'gcagctcca', 'gtacgatgaa', 'tcgacgccga', 'cagccttttg',
                 'acctctctag', 'gatcggcg', 'atagtcggtg', 'accgcgcggt', 'cgcaaaacaa', 'gtattgagtg', 'ctttcttcac',
                 'cgattttgag', 'taagaata', 'tagatggac', 'gctatcgac', 'ttcggcgccg', 'ttcaccacg', 'gcagctccac',
                 'ttccccaagg', 'agttgggtta', 'tcctaatttc', 'tagtgtttta', 'cgcgcgcgtt', 'tcggtcgacc', 'cgtgttgtca',
                 'atgataagtt', 'acttggccat', 'agttattttt', 'ctttaaccc', 'gtgggaccga', 'ctacccctca', 'taacccgagt',
                 'ttggtcgccg', 'cgcggccgcg', 'agcacactag', 'aattctcgtc', 'atcgaagga', 'accacccc', 'gctgggacgg',
                 'agcgcacctg', 'ccgtaacgga', 'cctactctcg', 'acggtaggc', 'gatccaacag', 'cgagagtggc', 'tgctagca',
                 'ttaataaa', 'agtttagagg', 'atggcgtatt', 'cgatgactt', 'catacactaa', 'agagacccct', 'gcaaaagggg',
                 'aactgggacc', 'tcgaatctcc', 'gttcgaacca', 'agggccttcc', 'tagtcgtctt', 'catttcccc', 'gatgatcatc',
                 'gggaggagag', 'ccgcctcgat', 'gcgaaatgc', 'tgccgcgcag', 'gacgcgtcg', 'aattaatt', 'ttcgtagaca',
                 'taagtgaggc', 'aaaaatcaa', 'ccgggcccgt', 'ttacctttt', 'tagcatcacg', 'catcttggg', 'tattcaaagg',
                 'ctctgaggac', 'ccaggtgct', 'cattggcact', 'gcctaaccct', 'tatctacg', 'ggtggagca', 'agataacaa',
                 'attttgaga', 'gccaggtgca', 'ggggatgatt', 'agacggccga', 'gaccggccg', 'ctgctgatg', 'gtactctcc',
                 'cgttgcaacg', 'cactaagaga', 'agtagctgaa', 'tgtaaggcc', 'gatccgcttt', 'tgaattctat', 'ctacttac',
                 'aatcactccg', 'ttgcaaat', 'cggagtccaa', 'ggatgtctgt', 'tccagtgtgg', 'gaaagctttc', 'ctggtgtgcc',
                 'agctcggtcg', 'gtagagccac', 'acacagaacc', 'ttattacgta', 'gtccccgttg', 'tttcacgg', 'ctttgcaa',
                 'tcaacgacag', 'tctttaga', 'cacctaaaag', 'gccgcaacta', 'tgccgtcttt', 'gccggccttt', 'gctgttggtt',
                 'attagtcgga', 'caagattttc', 'taatggctc', 'cgagtgccca', 'gaatatgcag', 'atgcagcaga', 'cggtatgttc',
                 'tagtatcttt', 'agacaggcct', 'cttcacgctt', 'caccccccca', 'aaaccgcaga', 'tcgtacga', 'gcaaatagtc',
                 'gaagatacca', 'caccgatta', 'gcactaagc', 'atatcacct', 'cgctcaattc', 'gaacccattc', 'tctcgccttg',
                 'ggccggcc', 'atgatcat', 'ggtagaccac', 'gataccactc', 'tcgtagtacc', 'tcatcgccg', 'cgcagggcct',
                 'ttgagcagat', 'accatggt', 'aacctgcagg', 'accagcgaag', 'ttatatagag', 'gcagccgtcg', 'gacacaggcg',
                 'ccatacgtg', 'ggggtttgag', 'ttaatcaac', 'cgatgcatcg', 'acagaaat', 'tagatctaa', 'atgttacaat',
                 'ggagttctga', 'ctttagaacg', 'ggctggggg', 'tgagacttgg', 'ctttaaacc', 'cctgggtat', 'atctttaga',
                 'tacgttgt', 'tcagtattga', 'gtcgtcaacc', 'atgcttagat', 'gggtcagga', 'ttctttttaa', 'tccttgacg',
                 'aacactatac', 'gaaaccgca', 'ggaagtcccg', 'caaccgctc', 'atcgatctgc', 'ggtttccctc', 'aattgtgtaa',
                 'attgccaa', 'gactctaatc', 'cattgcaatg', 'tacccctcat', 'agccgcttt', 'gcctttgt', 'cgaaaaggg',
                 'ggcaccggac', 'tagtcggtgg', 'gttcctgtca', 'agactaatat', 'tcgtgggacc', 'acattaatgt', 'cctcaccct',
                 'tacatcgcg', 'tcagtctgtc', 'gcttaaaac', 'ggcttaagcc', 'gagttcaccc', 'gctgagtctg', 'ctcggtgatt',
                 'aggcgcattt', 'tcattttgtt', 'cacgctatg', 'ctcaaggac', 'tcaaacatt', 'gggagacccc', 'ggatgaccgt',
                 'cgataaaatc', 'caagaccaag', 'gaccggtc', 'ccaactta', 'ggctgagtat', 'aacttactg', 'ccctgcaggg',
                 'ttgctatag', 'cgatcggagt', 'gcaataggtc', 'tgctactaaa', 'acgggtggtt', 'gttgaatttg', 'ggtaacatat',
                 'ggatgctc', 'cccagaacct', 'cggcggca', 'atctaggatc', 'gcgtggcgga', 'ttaacaacta', 'tcgcaggaca',
                 'tcactggcga', 'ctccagctca', 'ctgtgttctc', 'ctcaaccct', 'cattagctgc', 'aagttagcga', 'actgaagg',
                 'aggccttt', 'atcgatttgc', 'ccattaatgg', 'ccctcttca', 'tcggtcgacg', 'cttactgatt', 'tttgtcaga',
                 'cacactcctt', 'tacgcgta', 'acacgagac', 'ctcctctaaa', 'catcaaggct', 'gagtatatc', 'ggtggtttaa',
                 'cttctgaga', 'atcaaacgtt', 'ggagatcact', 'ttagaggccg', 'tcagtactga', 'tttggactcc', 'agaccttat',
                 'catgttccca', 'agtgcctctc', 'gaccggatgc', 'cattgccgaa', 'gtggaaacc', 'aaggctcgca', 'ttctttaga',
                 'tctgtgtaca', 'cgaacgttcg', 'tgcagcatgt', 'agagtagg', 'cctctaaata', 'aaaacactat', 'cccttttgc',
                 'attcgagcgg', 'tacccctgaa', 'gccaccggcg', 'gcgtcactgt', 'gacgactc', 'acggaaaaag', 'gaacatgttc',
                 'cgtcacctat', 'gggaacatct', 'taagcttta', 'gcagcgaatt', 'atatctgt', 'ggggaggata', 'ccctgcagta',
                 'ttgtcagttg', 'taaatactca', 'gcacgacgtt', 'aagcagagtt', 'aaatcgattt', 'acgagtac', 'ggatcggcg',
                 'gaatgcgcgc', 'acagtactt', 'ttcgcccagg', 'cgagacaaag', 'aagcagactt', 'ggccctcaga', 'ccttgcatga',
                 'ttgatctcgt', 'ctcgacatgc', 'caaagcgat', 'tgccgatca', 'cgggcaacac', 'aaatgtaagg', 'ccaaatccac',
                 'tttgaacac', 'acttatcatc', 'actcatgcag', 'cctcgtaggg', 'ttaaagggtc', 'ggatcggga', 'gtgaaacata',
                 'gaggagtgga', 'acatgtcat', 'atcatttagc', 'acgccttggg', 'ccataatgc', 'gcccttcaa', 'cgtccccagg',
                 'ggtctcgaa', 'aaccaccgag', 'gacgaggcag', 'taactagtta', 'cacagcgccc', 'cactcgccgc', 'tctccttggt',
                 'ccaagcatcg', 'aatgttacag', 'ccggcctacc', 'gtgcttacgg', 'aacacagggt', 'gacggaagtc', 'gcaatgtcaa',
                 'gagccttatc', 'gtgcacctct', 'acgcaggatc', 'ccaaagatt', 'aaaggtgagt', 'ccaggggatc', 'ctacatactt',
                 'ggagacccca', 'gccgcaaggg', 'tacgctggt', 'gaactcgact', 'caactaatc', 'ccaggtttaa', 'gtgagttcct',
                 'cgcgcgcgcg', 'ctgaactaa', 'ctcaatgccg', 'cacagtcttg', 'actccggtta', 'accggctcgg', 'tcctttgg',
                 'attacggat', 'tatgaatca', 'ataatgcg', 'agccgcttc', 'aggtgcaatc', 'tcccccaagg', 'ctccaagcg',
                 'tcagttgtag', 'ttgcattcga', 'accaccgact', 'ggtacctagc', 'cggtgcgctc', 'actgaaagtt', 'ccctgtaca',
                 'agtaagtt', 'tactcgacg', 'gtggtgataa', 'ctgacgtcag', 'aggcctttg', 'ggtccctcgg', 'gccgcgggga',
                 'cttacaaa', 'cgactggg', 'aactttttaa', 'tgagagcag', 'aaagcgcttt', 'tgtatccag', 'ctagcttaat',
                 'tgagtcgtct', 'gagctgcaca', 'ttaaagtctg', 'caagtgatct', 'gtgggctgtt', 'atttgctcta', 'gctttcatg',
                 'tcggtggagc', 'gttcccaacg', 'ccctctaaag', 'gcgacaagaa', 'tactgcggcc', 'tagcccagac', 'attgcataag',
                 'caatctaaaa', 'ctgcgtat', 'ccaggcgcc', 'tgttaactt', 'ctctgacat', 'caagggaac', 'aatcgatt', 'ttgtcgccg',
                 'gccaacgaat', 'aaattctttc', 'ccctactaaa', 'ctccctcgta', 'cctcggacac', 'ggcaccgagg', 'gcagacttg',
                 'cccgtggaga', 'gaccaacatc', 'tcctcagggt', 'aaggatca', 'ttctgcaaac', 'taaatactt', 'agcgatactc',
                 'cgttacgcac', 'aggagacatg', 'gcgaggtt', 'aatggcttac', 'gccatggc', 'ctcttcagg', 'tcgaaagccg',
                 'atatgata', 'cgttttacta', 'tgaacttgca', 'tacgtcga', 'aaaagacatt', 'ttaccagg', 'ggctatagcc',
                 'ttcttcacac', 'ggtctcccca', 'tgatggagct', 'accagctcaa', 'cactctggaa', 'ctccgacaca', 'caatcatt',
                 'ggcgttcg', 'atctgcacct', 'ccaaaaacgc', 'tcaaaggcct', 'gagcggcac', 'tcccgcccgt', 'tccagctca',
                 'tcgacgtcaa', 'ggcgagtac', 'tttccacgc', 'gccatatgat', 'atccggat', 'tcgccggtca', 'ccggcccca',
                 'tgaacaacgt', 'gtaagtgaga', 'gtctcagatc', 'ggagagtctc', 'tcagcggcga', 'cacggcccca', 'ccgaccattg',
                 'taagtttatc', 'gagccagccg', 'cgagcgcggc', 'aggtaccgac', 'accgcctcgg', 'catccgcgaa', 'gagtgatcca',
                 'ttggcactca', 'tcggctgctt', 'ggaattcc', 'taggtgaaaa', 'tacccatac', 'tatcacgatg', 'tgaagtcgg',
                 'gagatgacac', 'gtttcaccac', 'atatcaaaat', 'atgaatggaa', 'gtgtggacag', 'aagatatctt', 'cagggagaga',
                 'cggtaggatt', 'tattcaggct', 'tacctagaag', 'gtattactca', 'gcgacgtta', 'tgatgaggtg', 'taccgtta',
                 'gatcgatc', 'agagcacgtt', 'acagatggt', 'gatgtgcttc', 'cttatccccg', 'aaaacgcaga', 'cgactatttg',
                 'gatgataagt', 'cgcagaactc', 'cactaggaat', 'tgcaaaaact', 'gagacggccg', 'gcgacttccg', 'agaagtcgga',
                 'catcgctcag', 'cttatggcgt', 'gacaacgggc', 'ttactcac', 'tgccggatat', 'ctaaccgta', 'cacggttcg',
                 'ggagcaggaa', 'caaatacgag', 'caggataacc', 'ttcaattttt', 'aagcggtc', 'tatcacgcac', 'cgacgggcag',
                 'agtcgcgact', 'tcaaaacaag', 'atattttat', 'gctgttgga', 'gagtccaa', 'tccttggact', 'tggtactaga',
                 'aggagactg', 'gggcaggata', 'atgcagct', 'ggcacaactc', 'ggctctttgc', 'caagtgcctc', 'gatgactttg',
                 'gagaaagtc', 'atcgactacc', 'ttgtgggcac', 'gaatatttct', 'ataaattgtt', 'acggtctaat', 'ttgcaacagt',
                 'cagtccat', 'gatcggcgtt', 'attcgcgaat', 'ataacttccg', 'gccccctgca', 'ccagcttagt', 'aaacgtcagg',
                 'tgctccttcg', 'tgtcgcgagg', 'gtggcgttg', 'ccatcagct', 'aacgcgtt', 'aaccggttta', 'ttgagaaccc',
                 'gtagcgcac', 'ccatggaaat', 'cgactgggcc', 'gcgtcatgtt', 'aaagactgcg', 'gtcagtagct', 'ctgatagt',
                 'tcaagtacgg', 'tttggtatcc', 'gatgaaatg', 'ggggtctcc', 'tacttcttt', 'cgccggccga', 'gggaatggg',
                 'tattagctga', 'caagttaagt', 'ggcatgccag', 'ggacgcgtcg', 'gtgagtcaag', 'cggtcactgt', 'cttaccaggc',
                 'tcaatacatt', 'agccgtgcc', 'acacaaactc', 'gtgaatccca', 'ctcaaccc', 'gagcgctc', 'tgcgatgtt',
                 'agtcgccgct', 'aaggtctttt', 'gataacttct', 'caatattg', 'agctgaacag', 'gcagcaccgg', 'ctggctaata',
                 'tacggggc', 'ttgcagccgt', 'ccactgaat', 'cgatcactta', 'cattgatgcc', 'gagatcgga', 'gactatttg',
                 'tgaatatgtt', 'ctctctttgc', 'gcccctaggt', 'tgaacgtt', 'tctcgaaccc', 'tcttgcagcc', 'tgcttaccag',
                 'ggggcgaaga', 'gtgtgctcgc', 'taccattg', 'gggatccc', 'ccaggtgctg', 'gggtatcgtg', 'tccgtccaaa',
                 'agcgacgcac', 'aagcttaagc', 'aatatctcag', 'ctcgcgcac', 'cgccggcg', 'agcgtaatc', 'aagagttcca',
                 'agggtatttc', 'gtgcagctg', 'ttggagtca', 'tcggaagtc', 'gttataac', 'actgttgcgg', 'gcggggtagg',
                 'ccgcgcgcgg', 'gaggcctggc', 'tatataagct', 'ctccttgg', 'gtggaaacac', 'acttttaatc', 'atgaaccca',
                 'ggtaaggact', 'cttgtagg', 'aggtatatac', 'ggcaaacgtt', 'gtccccttgc', 'gtatgtgtac', 'cctgtggtga',
                 'tgccttgaa', 'caatcgtaac', 'agctccacc', 'catcttggtt', 'cctcgtgtgt', 'cggaccattc', 'aacagaagt',
                 'cgcgggatat', 'gagcgtcgat', 'agagagggt', 'gtaaggcctt', 'ctagacttgg', 'acaaatttgg', 'cggaattccg',
                 'aaggaatgg', 'taatcctagc', 'gtcgtctaac', 'gggcgaaatg', 'tcatactggt', 'gtaggcctag', 'ttctcaaaag',
                 'atgataagt', 'caaagcaagc', 'ggtttaccc', 'ccctagata', 'agacagtc', 'gttaaacagt', 'aaaccccgg',
                 'cagagtctg', 'ctgtggacag', 'cttgctctgc', 'ctgtacag', 'tacgctggta', 'caatagcagc', 'gctctgcgac',
                 'ttactatgtg', 'actacctaa', 'gctgcagccc', 'cttgacct', 'ccaaactact', 'cagtggtcgc', 'ccgcaccgt',
                 'cgcagtagg', 'ggggaattac', 'gtttttatga', 'cgatattact', 'ctttgaatgg', 'aggtttcgat', 'gtaccagca',
                 'acgccttaac', 'ttcagctttc', 'gggtcaacag', 'gattgttctg', 'gcagtcgag', 'gttcgcaagg', 'accatcatac',
                 'ctcctaggcc', 'tcgctatcg', 'ttagaatttg', 'actagataca', 'ctgggaatct', 'caatttcggc', 'gtgggccgt',
                 'gaatgccg', 'tattacgctg', 'tagcacgaac', 'aaagcttt', 'gatccgggga', 'gtgtgcgatg', 'cttgtatg',
                 'accctcata', 'gaggagactc', 'cggaggcacg', 'gtgaattcac', 'acaaggtcg', 'aaatcaaaag', 'cctgtgggtc',
                 'tcgagtgccc', 'caaacgcgtt', 'ctgcacatc', 'actgtttgtg', 'gattgagtag', 'atgttgatat', 'gcggaccaac',
                 'acaccccgta', 'atgacttt', 'tacggcagg', 'gatgaaatgc', 'ccatatctg', 'atctccgt', 'ctaattgctt',
                 'aatcagctgt', 'ttcgccatca', 'gcatttcgg', 'acccatgtac', 'cgccttgggg', 'tgtaaatgac', 'gctagggaaa',
                 'gaattccagg', 'gttgcttaac', 'tacacgac', 'acatcgcgg', 'gatgcgcaga', 'ttaacggtac', 'agaacgtcc',
                 'gaatcttgct', 'tgcctggc', 'catttttgcc', 'tgcggtgtcc', 'ctacgctcgg', 'accaccccgt', 'ttggtaaaa',
                 'ttgtgaccc', 'ttacatcctg', 'aagcaggggc', 'tctacagggt', 'gagcgaacgt', 'cgtaaaacta', 'ctggcctgtc',
                 'ggggtcagg', 'acccaaaaga', 'tcgagatcag', 'tcatatga', 'caggtgcggc', 'atatggtct', 'tgtgcctggc',
                 'tcacaccat', 'tagtgtttgg', 'ggtcgaag', 'atgaacccat', 'gtggaggagg', 'ccgtcagc', 'agcctagtg',
                 'taatgcccga', 'gacaggctga', 'gcagatgaa', 'gtgtatgat', 'ataaaacac', 'gagaattct', 'gcagggcctg',
                 'cgacgcgtcg', 'gcgcagggc', 'tagcgctgat', 'cgaccggcgg', 'tcgctaagcg', 'accggctact', 'tggaacggct',
                 'gttgcaaagt', 'cggacgctcg', 'agctcctag', 'gtcaaactga', 'ggggccacca', 'cctagattct', 'aggacccccc',
                 'tcgccgctgc', 'caccaata', 'gtccattgtg', 'gagatcatc', 'aaactacaga', 'cgagagtgaa', 'acgcacctg',
                 'gagcaggagc', 'cggttcacat', 'ataacgaacg', 'cggagtttag', 'cagagaacat', 'cccaacccca', 'atatggtgtc',
                 'taataggacc', 'ctaatgatac', 'cagcagggcc', 'ttgcatacta', 'cgtcgcca', 'tcaggcactg', 'gtatatac',
                 'caacatccg', 'agttgcatg', 'gtaaatcagg', 'gtcctggcc', 'ttgatgtt', 'gcacggcgtt', 'tcgcgcga',
                 'gagacgacca', 'gcgggagtat', 'ccgtctcc', 'aaagccgca', 'gggcagtgac', 'cacattat', 'taggtcctcg',
                 'caacttcgtg', 'cggcgtaaa', 'ttcgtctcg', 'gcggcaattc', 'tacgtaca', 'acacgcggat', 'acaaagcc',
                 'cggcgccgaa', 'ttatgggtaa', 'aaacatagac', 'cgataatctg', 'gtgatcccct', 'tcggcggca', 'caggtgctcg',
                 'atatcaatcc', 'acactgcccc', 'cggctcagcc', 'gcagagata', 'cgacacac', 'tttctttca', 'cttcatggag',
                 'cagaattctt', 'atttgtaatg', 'cgccagcttg', 'ataaggccga', 'cgaatgata', 'tgcgttatcc', 'tattcagcgt',
                 'gatgccttgc', 'accaggta', 'tgccaatact', 'catggtgacg', 'acatgggcac', 'tttgtcatg', 'tattatgaag',
                 'ccacggcac', 'cctcatcgct', 'acccaccggc', 'caatggcctg', 'gaggaccta', 'gagacttggt', 'gcaagtctga',
                 'agatactagt', 'ctcgccacg', 'tctttgtgt', 'aaggccggcc', 'cccctggag', 'atgatatgtc', 'ggggcacgcc',
                 'ccagcgtac', 'ttataataaa', 'tgcgggtaat', 'ggcatgcc', 'agacggccg', 'gcgttctac', 'atatttctgc',
                 'catttcatc', 'gactctaat', 'cttcacgtca', 'agctcctcga', 'aatgcctcc', 'ttgcttgcc', 'cgtctgcagg',
                 'tggtggtgcg', 'agtgggttat', 'tgtgccatct', 'cctaggcctg', 'gatggagctg', 'tcataaacgg', 'atcgggcatt',
                 'acgcgggaat', 'cgaactcggt', 'gccaatagtc', 'aggcactgtg', 'gaaactggaa', 'agagaggtac', 'atggggagac',
                 'gcctccttgg', 'cagtcgtt', 'gacatagcgt', 'gtctttctt', 'tttttggaca', 'caacgcaaat', 'ggccgacgcg',
                 'taatattaaa', 'ctgatcggcc', 'actggtattt', 'cttctgga', 'gtgagttttc', 'actaaccatg', 'cggtgcgcat',
                 'tcatttttcg', 'ccggaattgc', 'aagccggatg', 'tttcgacgtg', 'gtggacgaat', 'atcgattcgt', 'acccgtatag',
                 'accaccgaga', 'ggtaatatta', 'agacacgcaa', 'gggagtgat', 'aaatcccgcg', 'gcctggattt', 'ctgcccgtct',
                 'tataagcgt', 'gtcttgttta', 'cttttaacc', 'tgtcggctca', 'gctgaaagtg', 'ttataagaa', 'cgccgctatt',
                 'tccctactct', 'gatcaccggc', 'aggcagctac', 'gttttttcaa', 'ggatagaaa', 'gggggact', 'ccagcgta',
                 'ggactcctaa', 'gagctgcat', 'ggtggccacc', 'cccctaggtt', 'taacgcacac', 'gctcctaggc', 'gagtctcctc',
                 'tcaggcttg', 'gtagagctct', 'cacgctcgag', 'aaacttccgt', 'tcgcggatcc', 'acaatctat', 'ttcccgaggg',
                 'tgtatgcagt', 'gcatccgcag', 'ccaactaat', 'tcgttttact', 'ctgtgattca', 'cggccagtga', 'tgcggcgtgc',
                 'catggaaa', 'tcttgcgagc', 'tactcgtcca', 'gtattttct', 'gtcatgcagt', 'aggcggga', 'tccaatatc',
                 'ctagccgagt', 'tgactcca', 'cggcctctaa', 'tgtatcaaga', 'tgaacataac', 'tgttcactaa', 'gccaagcatg',
                 'cacccatgtt', 'aacaatccc', 'ctgcctcgg', 'gacctagaac', 'tacttatcg', 'acgcggccag', 'gcacgctctg',
                 'ttgtgatagg', 'agattggtcc', 'tagtgaggg', 'ccgttggcgg', 'ttgagttaaa', 'atagtagaga', 'cggtcaggta',
                 'gtctgagaac', 'ctgcagtat', 'gaaggatgcg', 'gcaatagtgg', 'gtcgggaagc', 'agggccctgt', 'tgatgttagt',
                 'taacaaactc', 'gtattcga', 'tgttcactct', 'taggcgctcc', 'cggtttatca', 'acagtgctc', 'gtttatct',
                 'gtgatatcga', 'taggctactg', 'atgcaagac', 'atctttaaa', 'cctgatcagg', 'cagctcca', 'cgatgtca',
                 'aaagactca', 'gagcgtgcag', 'cagaacccgt', 'gaatccaggg', 'acgtatttc', 'ggtttggact', 'agcttaacc',
                 'tcccctga', 'atcgccatt', 'aaataaaagt', 'gacctgcgg', 'cctcaaag', 'tgattgaa', 'taaggaattt', 'gccgattatc',
                 'ctagtcat', 'gaactcacga', 'tctgagacg', 'aagagacg', 'ccgtacgg', 'ggcctttg', 'gtagtgtg', 'gaacccat',
                 'aaatgcaaag', 'acttgagatg', 'gttacgtcgc', 'cccaagctca', 'gtagctgct', 'tgtcgttctc', 'gggataccat',
                 'tcggcaaggc', 'ttgtgcatg', 'ggtctcatt', 'tactccccac', 'tagctccac', 'gaaaatctga', 'ggcaaaggtc',
                 'aaagagacc', 'gttttatcag', 'taatgaaacg', 'ttcgtgagat', 'ttgcgcga', 'cttaacactc', 'tgtaaacatt',
                 'agtcgact', 'tacacccatc', 'acaacggg', 'cctttgaat', 'agtatagtta', 'atgaggaatc', 'aacccgggtt',
                 'aatgcacatg', 'gatctggtca', 'tcgatcctga', 'cagaggtcga', 'agactgactg', 'gggatgggga', 'cctactaaa',
                 'aaagagcatc', 'cagaagaaat', 'atgacgtgtc', 'gttcgcctaa', 'catgcgcatg', 'aagtaacgac', 'gctttgacgg',
                 'atgtggctca', 'ggcaatgaa', 'gctgatttgt', 'gtacagactt', 'atgcacctat', 'aggctgcctc', 'gttagctgag',
                 'acgccggccg', 'aggtatcaag', 'tgtagtgtta', 'tctttttgtc', 'atccttgtgt', 'cgcactcgac', 'agcttaatac']

import unittest

from src.genome.cache import Cache
from src.genome.sequence import Sequence
from src.segment.extender import Extender
from src.segment import seg_pool
import numpy as np
from time import time
from src.genome import seq_manager

class MockLoader(Loader):
    def _load_sequence_files(self):
        return None

    def _get_train_seq(self):
        return None

    def _get_test_seq(self):
        return None

class MockFileLabel:
    def __init__(self,):
        pass


class TestCache(unittest.TestCase):
    def setUp(self):
        self.sequence = Sequence('test_genome.fna', concatenate_nodes=False)
        seg_pool.segments = TEST_SEGMENTS * 6
        seg_pool.current_max_length = 10
        seg_pool.last_length = 8

        self.loader = MockLoader(MockFileLabel())
        self.loader.train_labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        self.loader.test_labels = np.array([0, 1])

        seq_manager.add_train_sequences([self.sequence] * 10)
        seq_manager.add_test_sequences([self.sequence] * 2)

    def test_get_count_from_seg_manager(self):
        start_time = time()
        o = self.sequence.get_count_from_seg_manager(seg_pool)
        print(time() - start_time)

        s = sum(o)
        print(s)

    def test_get_count_from_seg_manager_python_implementation(self):
        self.sequence._load_lib = lambda :None
        start_time = time()
        o = self.sequence.get_count_from_seg_manager(seg_pool)
        print(time() - start_time)

        s = sum(o)
        print(s)

    def test_get_dataset_from_pool(self):
        ray.init(num_cpus=12)
        start_time = time()
        self.loader.get_dataset_from_pool(False)
        print(time() - start_time)

        ray.shutdown()