void variant_file::output_windowed_weir_and_cockerham_fst(const parameters &params)
{
	int fst_window_size = params.fst_window_size;
	int fst_window_step = params.fst_window_step;
	vector<string> indv_files = params.weir_fst_populations;

	if ((fst_window_step <= 0) || (fst_window_step > fst_window_size))
		fst_window_step = fst_window_size;

	if (indv_files.size() == 1)
	{
		LOG.printLOG("Require at least two populations to estimate Fst. Skipping\n");
		return;
	}

	if ((meta_data.has_genotypes == false) | (N_kept_individuals() == 0))
		LOG.error("Require Genotypes in VCF file in order to output Fst statistics.");

	LOG.printLOG("Outputting Windowed Weir and Cockerham Fst estimates.\n");

	// First, read in the relevant files.
	vector< vector<bool> > indvs_in_pops;
	unsigned int N_pops = indv_files.size();
	indvs_in_pops.resize(N_pops, vector<bool>(meta_data.N_indv, false));
	vector<bool> all_indv(meta_data.N_indv,false);
	map<string, int> indv_to_idx;
	for (unsigned int ui=0; ui<meta_data.N_indv; ui++)
		if (include_indv[ui] == true)
			indv_to_idx[meta_data.indv[ui]] = ui;
	for (unsigned int ui=0; ui<N_pops; ui++)
	{
		ifstream indv_file(indv_files[ui].c_str());
		if (!indv_file.is_open())
			LOG.error("Could not open Individual file: " + indv_files[ui]);
		string line;
		string tmp_indv;
		stringstream ss;
		while (!indv_file.eof())
		{
			getline(indv_file, line);
			ss.str(line);
			ss >> tmp_indv;
			if (indv_to_idx.find(tmp_indv) != indv_to_idx.end())
			{
				indvs_in_pops[ui][indv_to_idx[tmp_indv]]=true;
				all_indv[indv_to_idx[tmp_indv]]=true;
			}
			ss.clear();
		}
		indv_file.close();
	}

	string CHROM; string last_chr = "";
	vector<string> chrs;
	vector<char> variant_line;
	entry *e = get_entry_object();

	// Calculate number of bins for each chromosome and allocate memory for them.
	// Each bin is a vector with four entries:
	// N_variant_sites: Number of sites in a window that have VCF entries
	// N_variant_site_pairs: Number of possible pairwise mismatches at polymorphic sites within a window
	// N_mismatches: Number of actual pairwise mismatches at polymorphic sites within a window
	// N_polymorphic_sites: number of sites within a window where there is at least 1 sample that is polymorphic with respect to the reference allele
	const vector< double > empty_vector(4, 0);	// sum1, sum2, sum3, count
	map<string, vector< vector< double > > > bins;
	double sum1=0.0, sum2 = 0.0;
	double sum3=0.0, count = 0.0;

	while(!eof())
	{
		get_entry(variant_line);
		e->reset(variant_line);
		N_entries += e->apply_filters(params);

		if(!e->passed_filters)
			continue;
		N_kept_entries++;

		e->parse_basic_entry(true);
		e->parse_full_entry(true);
		e->parse_genotype_entries(true);

		unsigned int N_alleles = e->get_N_alleles();

		if (e->is_diploid() == false)
		{
			LOG.one_off_warning("\tFst: Only using diploid sites.");
			continue;
		}

		vector<unsigned int> N_hom, N_het;
		vector<double> n(N_pops, 0.0);
		vector<vector<double> > p(N_pops, vector<double>(N_alleles,0.0));

		double nbar = 0.0;
		vector<double> pbar(N_alleles, 0.0);
		vector<double> hbar(N_alleles, 0.0);
		vector<double> ssqr(N_alleles, 0.0);
		double sum_nsqr = 0.0;
		double n_sum = 0.0;

		for (unsigned int i=0; i<N_pops; i++)
		{
			e->get_multiple_genotype_counts(indvs_in_pops[i], e->include_genotype, N_hom, N_het);

			for (unsigned int j=0; j<N_alleles; j++)
			{
				n[i] += N_hom[j] + 0.5*N_het[j];
				p[i][j] = N_het[j] + 2*N_hom[j];

				nbar += n[i];
				pbar[j] += p[i][j];
				hbar[j] += N_het[j];
			}
			for (unsigned int j=0; j<N_alleles; j++)
				p[i][j] /= (2.0*n[i]);	// diploid

			sum_nsqr += (n[i] * n[i]);
		}
		n_sum = accumulate(n.begin(),n.end(),0);
		nbar = n_sum / N_pops;

		for (unsigned int j=0; j<N_alleles; j++)
		{
			pbar[j] /= (n_sum*2.0); //diploid
			hbar[j] /= n_sum;
		}

		for (unsigned int j=0; j<N_alleles; j++)
		{
			for (unsigned int i=0; i<N_pops; i++)
				ssqr[j] += (n[i]*(p[i][j] - pbar[j])*(p[i][j] - pbar[j]));
			ssqr[j] /= ((N_pops-1.0)*nbar);
		}
		double nc = (n_sum - (sum_nsqr / n_sum)) / (N_pops - 1.0);

		vector<double> snp_Fst(N_alleles, 0.0);
		vector<double> a(N_alleles, 0.0);
		vector<double> b(N_alleles, 0.0);
		vector<double> c(N_alleles, 0.0);
		double r = double(N_pops);
		double sum_a = 0.0;
		double sum_all = 0.0;

		for(unsigned int j=0; j<N_alleles; j++)
		{
			a[j] = (ssqr[j] - ( pbar[j]*(1.0-pbar[j]) - (((r-1.0)*ssqr[j])/r) - (hbar[j]/4.0) )/(nbar-1.0))*nbar/nc;
			b[j] = (pbar[j]*(1.0-pbar[j]) - (ssqr[j]*(r-1.0)/r) - hbar[j]*( ((2.0*nbar)-1.0) / (4.0*nbar) ))*nbar / (nbar-1.0) ;
			c[j] = hbar[j] / 2.0;
			snp_Fst[j] = a[j]/(a[j]+b[j]+c[j]);

			if ((!isnan(a[j])) && (!isnan(b[j])) && (!isnan(c[j])))
			{
				sum_a += a[j];
				sum_all += (a[j]+b[j]+c[j]);
			}
		}
		double fst = sum_a/sum_all;
		if (!isnan(fst))
		{
			int pos = (int)e->get_POS();
			CHROM = e->get_CHROM();
			if (CHROM != last_chr)
			{
				chrs.push_back(CHROM);
				last_chr = CHROM;
			}

			int first = (int) ceil((pos - fst_window_size)/double(fst_window_step));
			if (first < 0)
				first = 0;
			int last = (int) ceil(pos/double(fst_window_step));
			for(int idx = first; idx < last; idx++)
			{
				if (idx >= (int)bins[CHROM].size())
					bins[CHROM].resize(idx+1, empty_vector);

				bins[CHROM][idx][0] += sum_a;
				bins[CHROM][idx][1] += sum_all;
				bins[CHROM][idx][2] += fst;
				bins[CHROM][idx][3]++;
			}

			sum1 += sum_a;
			sum2 += sum_all;
			sum3 += fst;
			count++;
		}
	}

	double weighted_Fst = sum1 / sum2;
	double mean_Fst = sum3 / count;

	LOG.printLOG("Weir and Cockerham mean Fst estimate: " + output_log::dbl2str(mean_Fst, 5) + "\n");
	LOG.printLOG("Weir and Cockerham weighted Fst estimate: " + output_log::dbl2str(weighted_Fst, 5) + "\n");

	string output_file = params.output_prefix + ".windowed.weir.fst";
	streambuf * buf;
	ofstream temp_out;
	if (!params.stream_out)
	{
		temp_out.open(output_file.c_str(), ios::out);
		if (!temp_out.is_open()) LOG.error("Could not open Fst Output file: " + output_file, 7);
		buf = temp_out.rdbuf();
	}
	else
		buf = cout.rdbuf();

	ostream out(buf);
	out << "CHROM\tBIN_START\tBIN_END\tN_VARIANTS\tWEIGHTED_FST\tMEAN_FST" << endl;
	for (unsigned int ui=0; ui<chrs.size(); ui++)
	{
		CHROM = chrs[ui];
		for (unsigned int s=0; s<bins[CHROM].size(); s++)
		{
			if ((bins[CHROM][s][1] != 0) && (!isnan(bins[CHROM][s][0])) && (!isnan(bins[CHROM][s][1])) && (bins[CHROM][s][3] > 0))
			{
				double weighted_Fst = bins[CHROM][s][0] / bins[CHROM][s][1];
				double mean_Fst = bins[CHROM][s][2] / bins[CHROM][s][3];

				out << CHROM << "\t"
				<< s*fst_window_step + 1 << "\t"
				<< (s*fst_window_step + fst_window_size) << "\t"
				<< bins[CHROM][s][3] << "\t"
				<< weighted_Fst << "\t" << mean_Fst << endl;
			}
		}
	}
	delete e;
}
