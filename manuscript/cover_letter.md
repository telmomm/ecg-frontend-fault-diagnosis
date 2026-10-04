# Cover letter

**To:** Editor-in-Chief, *IEEE Transactions on Instrumentation and Measurement*

**From:** Telmo Miguel-Medina, Electromechanical Engineering Department, University of Burgos, Burgos, Spain (tmiguel@ubu.es, ORCID 0009-0004-0654-6650)

**Date:** October 4, 2026

**Manuscript:** "Specification-Aware Fault Diagnosis of ECG Analog Front-Ends: Bridging Alternate Test and Machine-Learning Diagnosis for In-Service Self-Test" (regular paper, 8 pages, with supplementary material)

Dear Editor-in-Chief,

I submit the manuscript above for consideration as a regular paper.

**What the paper is about.** It asks whether an electrocardiograph can determine in service, with measurements it takes on itself, whether its analog front-end still meets the performance requirements of IEC 60601-2-25, which part has failed when it does not, and whether the cause is in the circuit or in the electrodes. The object of the study is a measurement system and its conformity to a measurement standard; machine learning is the tool, not the contribution.

**Why it belongs in TIM.** The work is about instrument self-test and fault diagnosis, and it responds directly to a paper published in this journal (Chen *et al.*, vol. 74, 2025, Art. no. 3549015), which showed that the benchmark circuits used in analog fault diagnosis condition its conclusions. The manuscript confirms that finding on a circuit of the application domain and proposes a different remedy: computing the ambiguity groups of the given circuit from its sensitivities and reporting the diagnosis at that level. The pass/fail decision is treated as a conformity assessment in the sense of JCGM 106, with escapes and false rejects as the consumer's and the producer's risk conditioned on the true state of the circuit. Classifier figures of merit are kept apart from statements about measurement accuracy, following the VIM.

**Main results.**

- On a front-end built around an integrated instrumentation amplifier, compliance is decided with 0.9 % of escapes at 7.8 % of false rejects; a limit test on the same measurements gives 22.5 % and 20.0 %.
- Defining a fault by component deviation, as the diagnosis literature does, instead of by loss of compliance raises false rejects from 1.6 % to 37 %.
- Diagnosis by ambiguity group reaches a macro F1 of 0.87 and holds for fault magnitudes absent from training; diagnosis by component does not.
- Three measurements taking 3 s come within one point of the balanced accuracy of the complete 97 s self-test.

**On the absence of experimental validation.** The study is based on simulation, and the manuscript says so and explains why. Labeling one case requires the eleven tests of the standard, and the campaign covers about 600 fault conditions with 200 Monte Carlo realizations each, most of which require a component to be altered; this is not feasible on hardware at that scale. The simulation also had to come first: which faults break compliance, which are visible to the self-test and which can be told apart must be known before a bench experiment on a meaningful subset can be designed. The measurement chain is modeled explicitly (converter noise, resolution, clipping, averaging), its parameters are swept in the robustness analysis, and the limitations are stated in the discussion. Hardware validation is the declared next step.

**Reproducibility.** The dataset (about 130,000 simulated cases with the value of eleven specifications each) is archived on Zenodo, doi:10.5281/zenodo.23134950, and the code that generates it and every table and figure is at https://github.com/telmomm/ecg-frontend-fault-diagnosis.

**Supplementary material.** A four-page file gives the tables and figures behind results that the main text reports in summary, and the schematic of the second circuit.

**EDICS.** Automation, Fault Diagnosis, Maintenance, and Testing; Medical, Biomedical, and Healthcare Instrumentation and Measurement.

**Declarations.**

- The manuscript is original, has not been published and is not under consideration elsewhere. No part of it has appeared in a conference, thesis or technical report.
- I am the sole author.
- The work received no funding.
- I have no conflict of interest to declare.
- AI-generated content was used and is disclosed in the Acknowledgment section of the manuscript: Claude (Anthropic) assisted in writing the code, producing the figures and drafting the text of all sections. I have reviewed and edited the content and take full responsibility for it.

Sincerely,

Telmo Miguel-Medina
