import os 
import gffutils
from Bio import SeqIO
from Bio.Seq import MutableSeq
import numpy as np
import h5py
import time
import argparse

donor_motif_counts = {}  # Initialize counts
acceptor_motif_counts = {}  # Initialize counts

def create_or_load_db(gff_file, db_file='gff.db'):
    """
    Create a gffutils database from a GFF file, or load it if it already exists.
    """
    if not os.path.exists(db_file):
        print("Creating new database...")
        db = gffutils.create_db(gff_file, dbfn=db_file, force=True, keep_order=True, merge_strategy='merge', sort_attribute_values=True)
    else:
        print("Loading existing database...")
        db = gffutils.FeatureDB(db_file)
    return db


def check_and_count_motifs(seq, labels, strand):
    """
    Check sequences for donor and acceptor motifs based on labels and strand,
    and return their counts in a dictionary.
    """    
    global donor_motif_counts, acceptor_motif_counts
    for i, label in enumerate(labels):
        if label in [1, 2]:  # Check only labeled positions
            if strand == '+':  # For forward strand
                motif = str(seq[i:i+2]) if i > 0 else ""  # Extract preceding 2-base motif
                if label == 2:
                    if motif not in donor_motif_counts:
                        donor_motif_counts[motif] = 0
                    donor_motif_counts[motif] += 1
                elif label == 1:
                    if motif not in acceptor_motif_counts:
                        acceptor_motif_counts[motif] = 0
                    acceptor_motif_counts[motif] += 1
            elif strand == '-':  # For reverse strand, after adjustment
                motif = str(seq[i:i+2]) if i < len(seq) - 1 else ""  # Extract following 2-base motif
                if label == 2:
                    if motif not in donor_motif_counts:
                        donor_motif_counts[motif] = 0
                    donor_motif_counts[motif] += 1    
                elif label == 1:
                    if motif not in acceptor_motif_counts:
                        acceptor_motif_counts[motif] = 0
                    acceptor_motif_counts[motif] += 1


def get_sequences_and_labels(db, fasta_file, output_dir, type, chrom_dict):
    """
    Extract sequences for each protein-coding gene, reverse complement sequences for genes on the reverse strand,
    and label donor and acceptor sites correctly based on strand orientation.
    """
    seq_dict = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
    fw_stats = open(f"{output_dir}stats.txt", "w")
    NAME = []      # Gene Name
    CHROM = []     # Chromosome
    STRAND = []    # Strand in which the gene lies (+ or -)
    TX_START = []  # Position where transcription starts
    TX_END = []    # Position where transcription ends
    SEQ = []       # Nucleotide sequence
    LABEL = []     # Label for each nucleotide in the sequence
    h5f = h5py.File(output_dir + f'datafile_{type}.h5', 'w')
    for gene in db.features_of_type('gene'):
        if gene.attributes["gene_biotype"][0] == "protein_coding":
            gene_id = gene.id
            gene_seq = seq_dict[gene.seqid].seq[gene.start-1:gene.end].upper()  # Extract gene sequence
            labels = [0] * len(gene_seq)  # Initialize all labels to 0
            transcripts = list(db.children(gene, featuretype='mRNA', order_by='start'))
            if len(transcripts) > 1:
                print(f"Gene {gene_id} has multiple transcripts: {len(transcripts)}")
            max_trans = transcripts[0]
            max_len = max_trans.end - max_trans.start + 1
            for transcript in transcripts:
                if transcript.end - transcript.start + 1 > max_len:
                   max_trans = transcript
                   max_len = transcript.end - transcript.start + 1
            exons = list(db.children(max_trans, featuretype='exon', order_by='start'))
            if len(exons) > 1:
                if gene.seqid in chrom_dict:
                    chrom_dict[gene.seqid] += 1
                    for i in range(len(exons) - 1):
                        # Donor site is one base after the end of the current exon
                        first_site = exons[i].end - gene.start + 1  # Adjusted for python indexing
                        # Acceptor site is at the start of the next exon
                        second_site = exons[i + 1].start - gene.start  # Adjusted for python indexing
                        # Ensure the sites are within the sequence bounds
                        if 0 <= first_site < len(labels):
                            if gene.strand == '+':
                                labels[first_site] = 2  # Mark donor site
                            elif gene.strand == '-':
                                labels[len(labels) - first_site-2] = 1  # Mark acceptor site
                        if 0 < second_site - 1 < len(labels):
                            if gene.strand == '+':
                                labels[second_site - 2] = 1  # Mark acceptor site
                            elif gene.strand == '-':
                                labels[len(labels) - second_site] = 2   # Mark donor site
                    if gene.strand == '-':
                        gene_seq = gene_seq.reverse_complement() # reverse complement the sequence
                    gene_seq = str(gene_seq.upper())
                    labels_str = ''.join(str(num) for num in labels)
                    NAME.append(gene_id)
                    CHROM.append(gene.seqid)
                    STRAND.append(gene.strand)
                    TX_START.append(str(gene.start))
                    TX_END.append(str(gene.end))
                    SEQ.append(gene_seq)
                    LABEL.append(labels_str)
                fw_stats.write(f"{gene.seqid}\t{gene.start}\t{gene.end}\t{gene.id}\t{1}\t{gene.strand}\n")
            check_and_count_motifs(gene_seq, labels, gene.strand)
    fw_stats.close()
    dt = h5py.string_dtype(encoding='utf-8')
    h5f.create_dataset('NAME', data=np.asarray(NAME, dtype=dt) , dtype=dt)
    h5f.create_dataset('CHROM', data=np.asarray(CHROM, dtype=dt) , dtype=dt)
    h5f.create_dataset('STRAND', data=np.asarray(STRAND, dtype=dt) , dtype=dt)
    h5f.create_dataset('TX_START', data=np.asarray(TX_START, dtype=dt) , dtype=dt)
    h5f.create_dataset('TX_END', data=np.asarray(TX_END, dtype=dt) , dtype=dt)
    h5f.create_dataset('SEQ', data=np.asarray(SEQ, dtype=dt) , dtype=dt)
    h5f.create_dataset('LABEL', data=np.asarray(LABEL, dtype=dt) , dtype=dt)
    h5f.close()


def print_motif_counts():
    global donor_motif_counts, acceptor_motif_counts
    print("Donor motifs:")
    for motif, count in donor_motif_counts.items():
        print(f"{motif}: {count}")
    print("\nAcceptor motifs:")
    for motif, count in acceptor_motif_counts.items():
        print(f"{motif}: {count}")


def main():
    # To-do: 
    # 4. let user provide the chromosome group for training and testing
    parser = argparse.ArgumentParser()
    parser.add_argument('--genome-fasta', '-g', type=str)
    parser.add_argument('--annotation-gff', '-a', type=str)
    parser.add_argument('--output-dir', '-o', type=str)
    args = parser.parse_args()
    fasta_file = args.genome_fasta
    gff_file = args.annotation_gff
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    db_file = f'{gff_file}_db'  # Replace with your database file path
    db = create_or_load_db(gff_file, db_file)
    TRAIN_CHROM_GROUP = {
        'chr2': 0, 'chr4': 0, 'chr6': 0, 'chr8': 0, 
        'chr10': 0, 'chr11': 0, 'chr12': 0, 'chr13': 0,
        'chr14': 0, 'chr15': 0, 'chr16': 0, 'chr17': 0, 
        'chr18': 0, 'chr19': 0, 'chr20': 0, 'chr21': 0, 
        'chr22': 0, 'chrX': 0, 'chrY': 0
    }
    TEST_CHROM_GROUP = {
        'chr1': 0, 'chr3': 0, 'chr5': 0, 'chr7': 0, 'chr9': 0
    }
    print("--- Processing ... ---")
    start_time = time.time()
    get_sequences_and_labels(db, fasta_file, output_dir, type="train", chrom_dict=TRAIN_CHROM_GROUP)
    get_sequences_and_labels(db, fasta_file, output_dir, type="test", chrom_dict=TEST_CHROM_GROUP)
    print_motif_counts()
    print(("--- %s seconds ---" % (time.time() - start_time)))

if __name__ == "__main__":
    main()