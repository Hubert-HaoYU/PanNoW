#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def run_cmd(cmd):
    try:
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def get_genome_basenames(input_dir, suffix):
    p = Path(input_dir)
    files = list(p.glob(f"*.{suffix}"))
    if not files:
        print(f"No files found with suffix '.{suffix}' in {input_dir}")
        sys.exit(1)
    return [f.stem for f in files]


def main():
    parser = argparse.ArgumentParser(description="Pangenome analysis pipeline.")
    parser.add_argument("-i", "--input_dir", required=True, help="Input directory of genome files")
    parser.add_argument("-x", "--suffix", required=True, help="Genome file suffix (e.g., 'fna', 'fasta')")
    parser.add_argument("-o", "--output_dir", required=True, help="Output directory")
    parser.add_argument("-j", "--threads", type=int, default=8, help="Max threads (default: 8)")
    parser.add_argument("-s", "--similarity", type=float, default=0.9, help="Protein clustering similarity (0.0–1.0, default: 0.9)")

    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    suffix = args.suffix.lstrip('.')
    threads = args.threads
    similarity = args.similarity

    if not input_dir.is_dir():
        print(f"Error: Input directory {input_dir} does not exist.")
        sys.exit(1)

    if output_dir.exists():
        response = input(f"Output directory {output_dir} exists! Overwrite? (y/N): ").strip().lower()
        if response != 'y':
            print("Aborted.")
            sys.exit(0)
        else:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    faa_dir = output_dir / "FAAs"
    gff_dir = output_dir / "GFFs"
    db_dir = output_dir / "diamondDBs"
    blast_dir = output_dir / "diamondResults"
    for d in [faa_dir, gff_dir, db_dir, blast_dir]:
        d.mkdir()

    basenames = get_genome_basenames(input_dir, suffix)
    print(f"Found {len(basenames)} genome files.")

    def run_prodigal(name):
        cmd = (
            f"prodigal -i '{input_dir}/{name}.{suffix}' "
            f"-a '{faa_dir}/{name}.faa' "
            f"-f gff -o '{gff_dir}/{name}.gff' -p meta"
        )
        return run_cmd(cmd)

    print("Running Prodigal...")
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(run_prodigal, name) for name in basenames]
        for future in as_completed(futures):
            if not future.result():
                print("Prodigal failed for some file. Exiting.")
                sys.exit(1)

    proteins_faa = output_dir / "proteins.faa"
    print("Merging all .faa files...")
    with open(proteins_faa, 'w') as outfile:
        for name in basenames:
            with open(faa_dir / f"{name}.faa", 'r') as infile:
                shutil.copyfileobj(infile, outfile)

    def make_diamond_db(name):
        cmd = f"diamond makedb --in '{faa_dir}/{name}.faa' -d '{db_dir}/{name}.dmnd'"
        return run_cmd(cmd)

    print("Building DIAMOND databases...")
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(make_diamond_db, name) for name in basenames]
        for future in as_completed(futures):
            if not future.result():
                print("DIAMOND makedb failed.")
                sys.exit(1)

    protein_nr = output_dir / "protein.nr"
    cdhit_similarity = min(1.0, max(0.0, similarity))
    word_size = 5 if cdhit_similarity >= 0.7 else 4 if cdhit_similarity >= 0.6 else 2
    cmd_cdhit = (
        f"cd-hit -i '{proteins_faa}' -o '{protein_nr}' "
        f"-c {cdhit_similarity:.2f} -n {word_size} -T {threads} -M 16000"
    )
    print("Running CD-HIT for protein clustering...")
    if not run_cmd(cmd_cdhit):
        print("CD-HIT failed.")
        sys.exit(1)

    def run_diamond_blast(name):
        cmd = (
            f"diamond blastp --query '{protein_nr}' "
            f"--db '{db_dir}/{name}.dmnd' "
            f"--evalue 1e-5 --id {int(similarity * 100)} "
            f"--sensitive --outfmt 6 --out '{blast_dir}/{name}.out' --threads 1"
        )
        return run_cmd(cmd)

    print("Running DIAMOND blastp...")
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(run_diamond_blast, name) for name in basenames]
        for future in as_completed(futures):
            if not future.result():
                print("DIAMOND blastp failed for some genome.")
                sys.exit(1)

    print("Generating presence-absence matrix and gene frequency...")
    matrix_file = output_dir / "presence_absence_matrix.csv"
    freq_file = output_dir / "gene_frequency.csv"

    import csv
    from collections import defaultdict

    def parse_out_file(filepath):
        gene_set = set()
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        qseqid = line.split('\t')[0]
                        gene_set.add(qseqid)
        except Exception as e:
            print(f"Error reading {filepath}: {e}", file=sys.stderr)
            return filepath, set()
        return filepath, gene_set

    out_files = sorted([blast_dir / f for f in os.listdir(blast_dir) if f.endswith('.out')])
    if not out_files:
        print("No .out files found!")
        sys.exit(1)

    genome_names = [f.name for f in out_files]
    total_genomes = len(genome_names)

    genome_gene_map = {}
    all_genes = set()

    with ThreadPoolExecutor(max_workers=threads) as executor:
        future_to_path = {executor.submit(parse_out_file, fp): fp for fp in out_files}
        for future in as_completed(future_to_path):
            filepath, genes = future.result()
            basename = filepath.name
            genome_gene_map[basename] = genes
            all_genes.update(genes)

    all_genes = sorted(all_genes)
    genome_names = sorted(genome_names)

    with open(matrix_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['gene'] + genome_names)
        for gene in all_genes:
            row = [gene] + ['1' if gene in genome_gene_map.get(genome, set()) else '0' for genome in genome_names]
            writer.writerow(row)

    gene_count = defaultdict(int)
    for genes in genome_gene_map.values():
        for gene in genes:
            gene_count[gene] += 1

    with open(freq_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['gene', 'occurrence_count', 'occurrence_ratio'])
        writer.writeheader()
        for gene in all_genes:
            count = gene_count[gene]
            ratio = count / total_genomes
            writer.writerow({'gene': gene, 'occurrence_count': count, 'occurrence_ratio': f"{ratio:.6f}"})

    core_genes = sum(1 for g in all_genes if gene_count[g] == total_genomes)
    unique_genes = sum(1 for g in all_genes if gene_count[g] == 1)

    print("\nPangenome analysis completed!")
    print(f"Outputs:")
    print(f"  - {matrix_file}")
    print(f"  - {freq_file}")


if __name__ == "__main__":
    main()
