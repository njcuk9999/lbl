"""The window's words, in English and in French, under the same keys
(tests/test_i18n.py checks that neither language lacks one).

OPTIONS below gives each PCA2D option a label and a help in both languages,
written from LBL's doc and the comments of its nominal.yaml. An option LBL
adds later is shown under its key, with LBL's own description (in English)
until it gets one here. The other (i) of the window are the help_* words.
"""

EN = {
    "title": "LBL",
    "banner_name": "line-by-line velocities",
    "tagline": "with or without the PCA2D precleaning: pick the targets, set, run, follow",
    "home": "Home",
    "eyebrow": "LBL · PCA2D",
    "intro": "The window writes an ordinary LBL wrap for the targets and options picked, runs it in the environment LBL is installed in, one run at a time, and shows its log as it goes. Every run keeps its wrap and its log in DATA_DIR/gui/runs.",
    "data_dir": "Data folder (DATA_DIR)",
    "data_dir_hint": "the folder holding science/<object>/ for each target",
    "scan": "Read the targets",
    "scanning": "reading headers: %s of %s spectra",
    "scan_done": "%s targets, %s spectra",
    "instrument": "Instrument",
    "source": "Data source",
    "python": "Python that runs the wraps",
    "tab_targets": "Targets",
    "tab_settings": "Settings",
    "tab_lbl": "LBL settings",
    "tab_pca2d": "PCA2D settings",
    "extra_title": "Kept from the loaded wrap",
    "extra_hint": "keys the window does not lay out, written back as they are into every wrap made from here",
    "load_title": "Load a wrap",
    "load": "Load",
    "load_hint": "the wrap is read, never run: its targets, templates, LBL steps and PCA2D options fill the tabs, and what the window does not lay out is kept as it is",
    "load_pick": "the wraps of the data folder",
    "loaded": "loaded: %s",
    "load_skipped": "computed in the file rather than written, so not read: %s",
    "tab_run": "Run",
    "tab_runs": "Runs",
    "tab_cleanup": "Cleanup",
    "tab_refs": "References",
    "refs_method": "The method",
    "refs_used": "Papers that used our codes (%s)",
    "refs_used_hint": "A list that is not exhaustive, up to date on %s.",
    "refs_backdrop": "The window's background",
    "refs_backdrop_hint": "It is made of blurred figures from these papers, all under a Creative Commons licence that allows it:",
    "col_target": "target",
    "col_instrument": "instrument",
    "col_files": "spectra",
    "col_nights": "nights",
    "col_snr": "median SNR",
    "col_exptime": "exposure (s)",
    "col_berv": "BERV (km/s)",
    "col_span": "BERV span",
    "col_dates": "dates",
    "col_teff": "Teff (K)",
    "all": "all",
    "none": "none",
    "ticked": "%s ticked of %s",
    "joint_note": "several targets ticked: one joint fit if 'joint' is on (Settings)",
    "plot_berv": "BERV coverage",
    "plot_berv_hint": "spectra per 3 km/s of barycentric velocity, stacked by target: the two frames separate only as far as the BERV moves. Every ticked target",
    "plot_snr": "SNR against BERV",
    "plot_snr_hint": "one dot per spectrum; three targets in colour, the others in grey. Every ticked target",
    "plot_time": "When",
    "plot_time_hint": "one row per target, one tick per spectrum; the bounds below keep only the spectra between them",
    "min_rjd": "from (RJD)",
    "max_rjd": "to (RJD)",
    "other_targets": "other targets",
    "no_points": "tick a target to see its spectra",
    "pca2d_on": "PCA2D precleaning",
    "pca2d_on_hint": "unticked, the wrap runs LBL alone",
    "basis_label": "%s (%s observer components, %s stars)",
    "group_model": "The fit",
    "group_bases": "Observer bases",
    "group_correction": "The correction",
    "group_runs": "What runs",
    "group_more": "More",
    "default": "default",
    "unset": "unset",
    "reset": "back to the default",
    "basis_none": "none: fit the observer components",
    "basis_group_kept": "kept, in %s",
    "basis_group_fits": "saved by the fits, in %s",
    "basis_keep": "keep",
    "basis_keep_tip": "move this basis into models/pca2d_%s_%s, where no cleanup deletes it; LBL reads it there first",
    "basis_other": "made with %s %s: LBL refuses it for this run",
    "per_target": "Per target",
    "lbl_setup": "Instrument and Python",
    "lbl_setup_hint": "The window reads them from the files of the ticked targets; choose otherwise here if you need to (a NIRPS mode, a source the files do not tell), and the line above says which is in force.",
    "setup_none": "Tick a target: the instrument and the data source are then read from its files.",
    "setup_auto": "Detected from the files of %s: %s.",
    "setup_unknown": "The files do not say which data source: %s is used, check it.",
    "setup_manual": "Chosen by hand: %s. The files of the ticked targets say %s.",
    "setup_reset": "Use the detection",
    "setup_mixed": "The ticked targets are of different instruments: tick one instrument only.",
    "setup_overruled": "%s, chosen by hand, cannot read the ticked files: %s, detected, is used.",
    "python_auto": "Python: the one this window runs with (default).",
    "python_manual": "Python: chosen by hand.",
    "comparison": "template",
    "comparison_self": "its own",
    "comparison_hint": "another star's LBL template (templates/) becomes the star's spectrum in PCA2D and LBL's template",
    "teff": "Teff (K)",
    "lbl_steps": "LBL",
    "skip_pca2d": "correction on disk",
    "skip_auto": "redo only if the request changed",
    "skip_keep": "keep it",
    "skip_redo": "redo it",
    "resproj": "temperature projection (DTEMP3500)",
    "sw_RUN_LBL_TELLUCLEAN": "telluric cleaning",
    "sw_RUN_LBL_TEMPLATE": "template",
    "sw_RUN_LBL_MASK": "mask",
    "sw_RUN_LBL_COMPUTE": "velocities",
    "sw_RUN_LBL_COMPILE": "rdb table",
    "overwrite_all": "rewrite the files already on disk",
    "help_overwrite_all": "Ticked: the template, the mask, the per-line velocities and the rdb tables are all made again, even those on disk (every SKIP_LBL_* False). Unticked: what is on disk is kept and only what is missing is made (SKIP_LBL_TEMPLATE, SKIP_LBL_MASK and SKIP_LBL_COMPUTE True), the rdb tables compiled again as always. Tick it when the spectra, the template or the mask changed. The boxes below set each step on its own.",
    "sw_SKIP_LBL_TEMPLATE": "keep the template on disk",
    "sw_SKIP_LBL_MASK": "keep the mask on disk",
    "sw_SKIP_LBL_COMPUTE": "keep the velocities on disk",
    "sw_SKIP_LBL_COMPILE": "keep the rdb on disk",
    "sw_REPORT": "report",
    "sw_PLANETS": "known planets (orbits and candidates)",
    "sw_SMALL_RV_SHIFT": "small velocity changes (template in one pass)",
    "run_name": "Run name",
    "run_name_auto": "auto",
    "wrap": "The wrap",
    "wrap_hint": "what will run, as it will be written to DATA_DIR/gui/runs/<run>/wrap.py; it can be edited before running",
    "check": "Rebuild from the settings",
    "save_wrap": "Save as wrap_<name>.py",
    "saved": "saved: %s",
    "errors": "Cannot run",
    "warnings": "Note",
    "queue_title": "Queue",
    "queue_empty": "nothing queued",
    "stop": "Stop",
    "cancel": "Take off the queue",
    "log": "Log",
    "log_top": "Top of the log (it stops following the run)",
    "step_goto": "Show where it starts in the log",
    "log_bottom": "Bottom of the log (it follows the run again)",
    "archive_log_top": "Top of the log (it stops following the search)",
    "archive_log_bottom": "Bottom of the log (it follows the search again)",
    "state_queued": "queued",
    "state_running": "running",
    "state_done": "done",
    "state_failed": "failed",
    "state_stopped": "stopped",
    "state_cancelled": "cancelled",
    "runs_window": "Runs from this window",
    "runs_reports": "Reports",
    "refresh": "Read again",
    "open": "open",
    "col_name": "name",
    "col_state": "state",
    "col_started": "started",
    "col_duration": "duration",
    "col_reports": "reports",
    "col_object": "object",
    "col_report": "report",
    "col_date": "date",
    "nothing": "nothing yet",
    "measure": "Measure again",
    "busy": "A run is going in this folder, so nothing can be deleted until it ends: %s",
    "measuring": "measuring...",
    "kind_scratch": "scratch: remade by the next run that needs it",
    "kind_rebuildable": "rebuildable: remade by running again",
    "kind_results": "results: kept",
    "kind_data": "spectra: never touched",
    "select_kind": "all of them",
    "select_kind_hint": "tick, or untick, everything of this kind",
    "select_type_hint": "tick, or untick, every one of this type",
    "selected_summary": "%s ticked, %s",
    "type_cache": "PCA2D cubes",
    "type_superseded": "replaced corrections",
    "type_corrected": "corrected spectra",
    "type_cleaned": "telluric-cleaned spectra",
    "type_lblrv": "per-exposure velocities",
    "type_template": "templates",
    "type_mask": "masks",
    "delete_selected": "Delete the ticked",
    "confirm_delete": "Delete %s items, %s? A link is removed as a link; what it points to stays.",
    "deleted": "%s deleted",
    "total": "total",
    "what_cache": "PCA2D's registered cubes, remade by the next fit",
    "what_superseded": "a correction replaced by a newer one, and what LBL made from it",
    "what_spectra": "the spectra of a target",
    "what_corrected": "corrected spectra (remade from the fit; tar them first if they are someone's)",
    "what_cleaned": "telluric-cleaned spectra",
    "what_lblrv": "LBL's per-exposure velocities (about ten minutes to remake)",
    "what_template": "an LBL template",
    "what_mask": "an LBL mask",
    "what_rdb": "LBL's velocity tables",
    "what_reports": "the reports",
    "what_components": "observer bases and stars' components",
    "what_models": "LBL's models: stellar models, TAPAS, vetted templates, kept PCA2D bases",
    "lbl_says": "LBL's own description:",
    "help_data_dir": "The LBL data folder (DATA_DIR). Each target's spectra are in science/<object>/; LBL writes its templates, masks, per-exposure velocities (lblrv), tables (lblrdb) and reports beside them.\n\n'Read the targets' reads every spectrum's header once (SNR, BERV, date, Teff) and keeps the result, so reading again later is quick.",
    "help_instrument": "The LBL instrument, and its mode, the spectra come from. It sets how LBL reads the files, the wavelength domain and the resolution. Guessed from the headers when the targets are read; NIRPS_HE and NIRPS_HA are NIRPS's two modes.",
    "help_source": "The pipeline that reduced the spectra (DATA_SOURCE: APERO, ESO, CADC...). It sets the file format LBL expects. A PCA2D basis made from another source cannot be used here.",
    "help_python": "The Python that runs the wraps: that of the conda environment LBL (its lbl-pca2d branch) is installed in. By default, the Python this window runs with.",
    "help_pca2d_on": "The precleaning: PCA2D separates what moves with the star from what stays fixed on the detector (telluric residuals, airglow, detector features), divides the latter out of the spectra, and LBL then measures the corrected spectra, science/<object>_pca2d/. The delivered spectra are never written.\n\nUnticked, the wrap runs LBL alone, on the delivered spectra.",
    "help_skip_pca2d": "What to do with a correction already on disk (SKIP_LBL_PCA2D).\n\n'redo only if the request changed': kept if the settings, the input files and LBL's nominal.yaml are those it was made with, made again otherwise. 'keep it': kept even if they changed, which a warning says. 'redo it': made again in every case; the previous one is moved aside, not deleted.",
    "help_comparison": "The template LBL measures each exposure against (OBJECT_COMPARISON).\n\n'its own': built from the target's own spectra. Another star's template (from templates/, of a similar spectral type) suits a target with few spectra or little BERV coverage, whose own template would be noisy or carry telluric residuals. With PCA2D, it is also the star's spectrum in the fit.",
    "help_teff": "The star's effective temperature (OBJECT_TEFF), in K. LBL uses it to pick the PHOENIX model behind the line mask and the systemic velocity. Read from the headers when they have it (grey), else asked of SIMBAD by the target's name (the median of the Teff it holds, those far from the star's spectral type left out, else its spectral type's), said under the field; a value typed here takes its place.",
    "help_resproj": "Adds a DTEMP3500 column to the rdb tables: each exposure projected on the derivative of a 3500 K spectrum with respect to temperature, read as a change of effective temperature in K. An activity indicator (spots, faculae) for M dwarfs, to set beside the velocities.",
    "help_bounds": "Only the spectra between these dates enter the PCA2D fit and its correction. In RJD, BJD - 2 400 000, as LBL's tables (one day apart, a bound at an integer falls at noon UT, between nights in the Americas). Empty: no bound. The plots on the left grey out what the bounds leave out.",
    "help_run_name": "The run's name: its folder, DATA_DIR/gui/runs/<name>/, keeps the wrap as it ran, its whole log and run.json (when, how it ended, the reports it wrote). 'auto' makes it from the targets and the settings.",
    "help_wrap": "The wrap that will run: an ordinary LBL script, which also runs by hand as it is. It follows what is ticked in the other tabs until you edit it here; 'Rebuild from the settings' starts it again from them. 'Save' writes it in the data folder as wrap_<name>.py.",
    "help_load": "Read a wrap back into the window: it is parsed, never run. Its targets, templates, LBL steps and PCA2D options fill the tabs; what the window does not lay out (INPUT_FILE, a PCA2D section.key...) is kept and written back into every wrap made from it, so that a wrap loaded and saved again loses nothing.",
    "help_sw_RUN_LBL_TELLUCLEAN": "LBL's own telluric correction (a TAPAS model fitted to each spectrum), in two passes: a first one blind, a template made of its cleaned spectra, then a second one of the spectra as delivered with the star's lines divided out by that template, so that they do not bias the fit, and the template made again from them. Written to science/<object>_tc/, which every later step then uses. For spectrographs whose pipeline delivers no telluric correction (HARPS, ESPRESSO, ESPaDOnS...); off for APERO's NIRPS and SPIRou spectra, already corrected. With PCA2D, the precleaning runs on the cleaned spectra.",
    "help_sw_RUN_LBL_TEMPLATE": "Build the star's template: all its spectra shifted into the star's frame and combined into one high-SNR spectrum (templates/). Every line is measured against it. Only for a target measured against its own template; another star's is read as it is.",
    "help_sw_RUN_LBL_MASK": "Build the line list (masks/) from the template: each line's position and sign, the domains the velocities are measured over, and the star's systemic velocity.",
    "help_sw_RUN_LBL_COMPUTE": "Measure the velocity of every line in every exposure against the template (lblrv/, one file per exposure). The longest of LBL's steps, typically about ten minutes.",
    "help_sw_RUN_LBL_COMPILE": "Combine each exposure's per-line velocities into one velocity and its error, with the indicators (d2v, d3v, DTEMP...), into the rdb tables of lblrdb/: lbl_ per exposure, lbl2_ per night.",
    "help_sw_SKIP_LBL_TEMPLATE": "Keep a template already on disk rather than build it again. Untick it when the spectra changed (more of them, another correction): the velocities must be measured against a template made from the same spectra. With PCA2D, a new correction moves the old products aside by itself.",
    "help_sw_SKIP_LBL_MASK": "Keep a mask already on disk rather than build it again. Untick it when the template was made again.",
    "help_sw_SKIP_LBL_COMPUTE": "Do not measure again an exposure whose per-line velocities are on disk: only the new exposures are measured. Untick it when the template or the mask changed, or old and new velocities get mixed in one table.",
    "help_sw_SKIP_LBL_COMPILE": "Keep the rdb tables on disk. Off by default: compiling takes seconds and must include every new exposure.",
    "help_sw_REPORT": "Write the PDF report in reports/: the velocities, their periodograms, the indicators and, with PCA2D, its own pages first (the river plots, the components, the velocities before and after).",
    "help_sw_PLANETS": "Use the star's known planets and candidates (LBL's planet catalogs, and DATA_DIR/personal_planet_catalog.yaml): the template is built on the BERV plus their Keplerian velocity, and the report fits their K at their periods. The velocities reported stay corrected for the BERV only.",
    "help_sw_SMALL_RV_SHIFT": "Ticked (the default): the star's velocity is taken as constant, and its template is made in one pass, every spectrum shifted by its BERV alone. Right when the velocity varies by less than about 100 m/s over the spectra (MAX_CONVERGENCE_TEMPLATE_RV): planets of a few to a few tens of m/s, activity; it saves the template's convergence passes. Unticked: each spectrum's velocity is measured against the template and the template made again, up to 5 times, until the shifts agree to 100 m/s; for a binary, a brown dwarf or a hot Jupiter (HD 189733's K is 205 m/s), whose lines would otherwise be smeared in the template. With known planets ticked and the star's orbits in the catalogs, the template follows the orbits instead. Unticked, PCA2D's star frame follows the velocities the template measured. Not PCA2D's velocity_term, a velocity per exposure fitted inside the PCA2D fit, which changes no template.",
    "launch": "Run",
    "launch_running": "Running · %s",
    "launch_more": "A run is going: a click now starts this one when it ends",
    "instrument_followed": "instrument set to %s, that of the ticked targets",
    "overwrite_q": "%s exists: overwrite it?",
    "err_no_folder": "the data folder %s is not there",
    "err_no_instrument": "no instrument chosen",
    "err_no_target": "no target ticked",
    "err_unknown_instrument": "LBL does not know the instrument %s",
    "err_no_source": "%s has no data source %s (LBL knows %s)",
    "err_wrong_instrument": "%s holds %s spectra, and the instrument chosen is %s: choose the instrument of its spectra, or untick it",
    "warn_no_teff": "no effective temperature for %s: LBL gets %s K (it only picks the stellar model of the mask)",
    "simbad_ask": "Ask SIMBAD for the Teff again",
    "simbad_asking": "SIMBAD...",
    "simbad_failed": "SIMBAD did not answer",
    "simbad_unknown": "SIMBAD does not know this name",
    "simbad_none": "SIMBAD: %s, no Teff",
    "simbad_teff": "SIMBAD: %s K (%s measurements)",
    "simbad_teff_some": "SIMBAD: %s K (%s measurements of %s)",
    "simbad_why": "%s for SIMBAD, of spectral type %s. The median of the Teff it holds, those more than a fifth from its spectral type's left out (Pecaut & Mamajek 2013).",
    "simbad_teff_spt": "SIMBAD: %s K, from its spectral type %s",
    "simbad_use": "use it",
    "err_no_comparison": "%s: the template %s is neither in templates/ nor can be made from spectra in science/",
    "err_pca2d_kind": "PCA2D %s: %s is not a %s",
    "err_no_basis": "no observer basis %s in the subfolders of %s, nor in %s",
    "err_basis_other": "the basis %s was made with %s %s, this run is %s %s: LBL refuses it",
    "err_pca2d": "PCA2D: %s",
    "basis_dropped": "the frozen basis %s is not one of %s %s: left out",
    "plot_vtot": "SNR against v_tot, the star's velocity in the observer's frame",
    "plot_vtot_hint": "one dot per spectrum, against v_tot = v_sys - BERV (composed relativistically), where the star's lines fall against the tellurics; the shaded band, |v_tot| < 4 km/s, is where they sit on them. The targets whose systemic velocity is known (SIMBAD, else the headers); every target is in the BERV plots",
    "help_vtot": "Where the star's lines fall on the detector against the telluric lines: its systemic velocity less the BERV, for each spectrum.\n\nThe systemic velocity is SIMBAD's, else the headers' (APERO's PP_RV when its source is a reference, or ESO's target velocity); a target without one is left out of this plot, and every target is in the BERV plots. A star whose v_tot stays within a few km/s of 0 has its lines on the same tellurics all year: telluric residuals then move with it and the two frames are hard to separate, the case for a frozen observer basis.",
    "no_vsys": "no systemic velocity in the headers of the ticked targets",
    "n_spectra": "%s spectra",
    "tellu_off": "off for NIRPS and SPIRou: their spectra come telluric-corrected by APERO",
    "vetted_group": "LBL's vetted templates",
    "vetted_label": "%s (%s K, %s spectra)",
    "best_match": "best vetted template",
    "best_match_all": "Best vetted template for all",
    "best_match_none": "%s: no vetted template for %s K (2900-4300 K only)",
    "err_vetted_unknown": "%s is not one of LBL's vetted templates for %s",
    "help_vetted": "LBL's vetted templates: for each 100 K from 3000 to 4200 K and each instrument, the template of the star with the most spectra among slow rotators (a published rotation period, or narrow lines) that are not young stars. One is downloaded from LBL's depot the first time a run names it (VETTED_<star>).\n\n'best vetted template' picks, for a target between 2900 and 4300 K, the vetted template of the nearest temperature: a target with few spectra or little BERV coverage is then measured against a template of hundreds of spectra.",
    "steps_title": "Steps",
    "steps_planned": "as planned from the wrap; they tick off as the run goes",
    "steps_of": "steps of %s",
    "help_steps": "The steps of the run from end to end, as the wrap plans them: for each target, LBL on the delivered spectra (when PCA2D measures them first), PCA2D's cube, fit, figures and correction, then LBL on the corrected spectra and the report. Each one ticks off when the log reaches the next; the one going is outlined, a failed one is red.",
    "step_telluclean": "telluric cleaning",
    "step_telluclean_1": "telluric cleaning 1, blind",
    "step_template_tc": "template of the cleaned spectra",
    "step_telluclean_2": "telluric cleaning 2, divided by that template",
    "step_template_tc2": "template of the second cleaning",
    "step_template": "template",
    "step_mask": "mask",
    "step_compute": "velocities",
    "step_compile": "rdb table",
    "step_report": "report",
    "step_pca2d_cube": "PCA2D cube",
    "step_pca2d_fit": "PCA2D fit",
    "step_pca2d_figures": "PCA2D figures",
    "step_pca2d_correct": "PCA2D correction",
    "no_data_dir": "Choose a data folder (DATA_DIR) above: the folder holding science/<object>/ for each target.",
    "no_science": "%s does not exist. A data folder holds one folder per target in science/: science/<object>/, with that target's spectra.",
    "empty_science": "%s holds no target folder. Each target is a folder science/<object>/ with its spectra.",
    "read_of": "headers read: %s of %s",
    "err_mixed_title": "Two instruments ticked.",
    "err_mixed": "A run measures one instrument's spectra (%s): untick the targets of one of them.",
    "err_vetted_missing": "%s is not on LBL's depot yet: put %s in %s, or wait for the depot to have it",
    "warn_vetted_unknown": "LBL's depot does not answer: %s will be downloaded when the run needs it, if it can be",
    "sub_local": "My targets",
    "sub_archive": "Archive search",
    "archive_title": "Spectra in the public archives",
    "archive_target": "Targets (several: separate them with commas, e.g. GJ 1, GJ 2)",
    "archive_instruments": "Instruments",
    "archive_search": "Search the archives",
    "archive_fetch": "Fetch the ticked (%s)",
    "archive_fetch_none": "Tick what to fetch",
    "archive_star": "%s, Teff %s K",
    "archive_searching": "searching",
    "archive_found": "%s spectra",
    "archive_failed": "no answer",
    "archive_stopped": "stopped",
    "archive_skipped": "out of reach",
    "archive_search_stop": "Stop the search",
    "archive_log": "What the search and the fetch are doing",
    "archive_nothing": "the archives asked have no public spectra of %s that LBL reads",
    "archive_progress": "%s: %s of %s spectra, %s not fetched",
    "archive_left": "about %s left",
    "archive_took": "done in %s",
    "archive_waiting": "waiting for this archive's other fetches",
    "archive_done": "fetched into %s: the targets are in My targets",
    "archive_big": "%s to download into %s: go on?",
    "archive_unavailable": "the archive search cannot be used here: %s",
    "col_reader": "LBL reader",
    "col_archive": "archive",
    "col_have": "here",
    "col_size": "to fetch",
    "col_bytes": "size",
    "col_folder": "folder",
    "err_archive_source": "%s was fetched for %s %s: choose that instrument and data source",
    "err_layout_source": "%s holds spectra laid out for the data source %s, and the one chosen is %s: choose %s",
    "help_archive": "The public spectra that LBL reads, of one star or of several (separate their names with commas), in the archives the window knows (lbl.gui.fetch, carried into LBL from koloa): SPIRou's t.fits at the CADC, NIRPS, HARPS and ESPRESSO at ESO, HARPS-N at the TNG, SOPHIE at the OHP. Each name is resolved by SIMBAD (its name, its coordinates and its Teff are shown under the field), then every star is asked of every archive at once, two requests at a time in one archive. An archive whose telescope never sees the star (less than 20 degrees above its horizon at best) is not asked, said 'out of reach'; one that does not answer is said, and Stop the search stops waiting for it.\n\nThe ticked rows are fetched into DATA_DIR/science/<star>/ (calibrations in DATA_DIR/calib/), four files at a time, the instruments at once, nothing there moved or overwritten; a folder that holds another reader's spectra gets a folder of its own. Each folder keeps what it came from, and the wrap takes the data source and the file pattern from it. The log under the results says what each archive answered and why a file was not fetched.",
    "refresh_targets": "Refresh",
    "lang_switch": "FR",
    "theme_light": "Light",
    "theme_dark": "Dark",
    "tile_targets": "Ticked targets",
    "tile_folder": "of %s in the folder, %s spectra",
    "tile_spectra": "Their spectra",
    "tile_nights": "Their nights",
    "tile_snr": "median SNR %s",
    "tile_berv": "Their BERV coverage",
    "tile_none": "nothing ticked",
    "close": "close",
    "yes": "yes",
    "no": "no",
}

FR = {
    "title": "LBL",
    "banner_name": "vitesses raie par raie",
    "tagline": "avec ou sans le pré-nettoyage PCA2D : choisir les cibles, régler, lancer, suivre",
    "home": "Accueil",
    "eyebrow": "LBL · PCA2D",
    "intro": "La fenêtre écrit un wrap LBL ordinaire pour les cibles et les options choisies, le lance dans l'environnement où LBL est installé, un run à la fois, et en affiche le journal au fil du calcul. Chaque run garde son wrap et son journal dans DATA_DIR/gui/runs.",
    "data_dir": "Dossier de données (DATA_DIR)",
    "data_dir_hint": "le dossier qui contient science/<objet>/ pour chaque cible",
    "scan": "Lire les cibles",
    "scanning": "lecture des en-têtes : %s spectres sur %s",
    "scan_done": "%s cibles, %s spectres",
    "instrument": "Instrument",
    "source": "Source des données",
    "python": "Python qui lance les wraps",
    "tab_targets": "Cibles",
    "tab_settings": "Réglages",
    "tab_lbl": "Réglages LBL",
    "tab_pca2d": "Réglages PCA2D",
    "extra_title": "Gardé du wrap chargé",
    "extra_hint": "des clés que la fenêtre n'affiche pas, réécrites telles quelles dans chaque wrap fait d'ici",
    "load_title": "Charger un wrap",
    "load": "Charger",
    "load_hint": "le wrap est lu, jamais exécuté : ses cibles, ses templates, les étapes de LBL et les options PCA2D remplissent les onglets, et ce que la fenêtre n'affiche pas est gardé tel quel",
    "load_pick": "les wraps du dossier de données",
    "loaded": "chargé : %s",
    "load_skipped": "calculé dans le fichier plutôt qu'écrit, donc pas lu : %s",
    "tab_run": "Lancer",
    "tab_runs": "Runs",
    "tab_cleanup": "Ménage",
    "tab_refs": "Références",
    "refs_method": "La méthode",
    "refs_used": "Articles ayant utilisé nos codes (%s)",
    "refs_used_hint": "Liste non exhaustive, à jour au %s.",
    "refs_backdrop": "Le fond de la fenêtre",
    "refs_backdrop_hint": "Il est fait de figures floues tirées de ces articles, tous sous une licence Creative Commons qui le permet :",
    "col_target": "cible",
    "col_instrument": "instrument",
    "col_files": "spectres",
    "col_nights": "nuits",
    "col_snr": "SNR médian",
    "col_exptime": "pose (s)",
    "col_berv": "BERV (km/s)",
    "col_span": "étendue du BERV",
    "col_dates": "dates",
    "col_teff": "Teff (K)",
    "all": "toutes",
    "none": "aucune",
    "ticked": "cochées : %s sur %s",
    "joint_note": "plusieurs cibles cochées : un seul fit joint si « joint » est activé (Réglages)",
    "plot_berv": "Couverture en BERV",
    "plot_berv_hint": "spectres par tranche de 3 km/s de vitesse barycentrique, empilés par cible : les deux référentiels ne se séparent qu'autant que le BERV varie. Toutes les cibles cochées",
    "plot_snr": "SNR en fonction du BERV",
    "plot_snr_hint": "un point par spectre ; trois cibles en couleur, les autres en gris. Toutes les cibles cochées",
    "plot_time": "Quand",
    "plot_time_hint": "une ligne par cible, un trait par spectre ; les bornes ci-dessous ne gardent que les spectres entre elles",
    "min_rjd": "du (RJD)",
    "max_rjd": "au (RJD)",
    "other_targets": "autres cibles",
    "no_points": "cochez une cible pour voir ses spectres",
    "pca2d_on": "Pré-nettoyage PCA2D",
    "pca2d_on_hint": "décoché, le wrap lance LBL seul",
    "basis_label": "%s (%s composantes de l'observateur, %s étoiles)",
    "group_model": "Le fit",
    "group_bases": "Bases de l'observateur",
    "group_correction": "La correction",
    "group_runs": "Ce qui tourne",
    "group_more": "Autres",
    "default": "défaut",
    "unset": "non défini",
    "reset": "revenir au défaut",
    "basis_none": "aucune : ajuster les composantes de l'observateur",
    "basis_group_kept": "gardées, dans %s",
    "basis_group_fits": "enregistrées par les fits, dans %s",
    "basis_keep": "garder",
    "basis_keep_tip": "ranger cette base dans models/pca2d_%s_%s, où aucun ménage ne l'efface ; LBL l'y lit en premier",
    "basis_other": "faite avec %s %s : LBL la refuse pour ce run",
    "per_target": "Par cible",
    "lbl_setup": "Instrument et Python",
    "lbl_setup_hint": "La fenêtre les lit dans les fichiers des cibles cochées ; choisissez autre chose ici au besoin (un mode de NIRPS, une source que les fichiers ne disent pas), et la ligne ci-dessus dit lequel est en vigueur.",
    "setup_none": "Cochez une cible : l'instrument et la source de données sont alors lus dans ses fichiers.",
    "setup_auto": "Détectés d'après les fichiers de %s : %s.",
    "setup_unknown": "Les fichiers ne disent pas quelle source de données : %s est utilisée, vérifiez-la.",
    "setup_manual": "Choisis à la main : %s. Les fichiers des cibles cochées disent %s.",
    "setup_reset": "Revenir à la détection",
    "setup_mixed": "Les cibles cochées sont d'instruments différents : n'en cochez qu'un.",
    "setup_overruled": "%s, choisi à la main, ne peut pas lire les fichiers cochés : %s, détecté, est utilisé.",
    "python_auto": "Python : celui avec lequel tourne cette fenêtre (défaut).",
    "python_manual": "Python : choisi à la main.",
    "comparison": "template",
    "comparison_self": "le sien",
    "comparison_hint": "le template LBL d'une autre étoile (templates/) devient le spectre de l'étoile dans PCA2D et le template de LBL",
    "teff": "Teff (K)",
    "lbl_steps": "LBL",
    "skip_pca2d": "correction sur disque",
    "skip_auto": "refaite seulement si la demande a changé",
    "skip_keep": "la garder",
    "skip_redo": "la refaire",
    "resproj": "projection en température (DTEMP3500)",
    "overwrite_all": "réécrire les fichiers déjà sur le disque",
    "help_overwrite_all": "Cochée : le template, le masque, les vitesses par raie et les tables rdb sont tous refaits, même ceux qui sont sur le disque (tous les SKIP_LBL_* à False). Décochée : ce qui est sur le disque est gardé et seul ce qui manque est fait (SKIP_LBL_TEMPLATE, SKIP_LBL_MASK et SKIP_LBL_COMPUTE à True), les tables rdb étant recompilées comme toujours. À cocher quand les spectres, le template ou le masque ont changé. Les cases dessous règlent chaque étape séparément.",
    "sw_RUN_LBL_TELLUCLEAN": "nettoyage tellurique",
    "sw_RUN_LBL_TEMPLATE": "template",
    "sw_RUN_LBL_MASK": "masque",
    "sw_RUN_LBL_COMPUTE": "vitesses",
    "sw_RUN_LBL_COMPILE": "table rdb",
    "sw_SKIP_LBL_TEMPLATE": "garder le template sur disque",
    "sw_SKIP_LBL_MASK": "garder le masque sur disque",
    "sw_SKIP_LBL_COMPUTE": "garder les vitesses sur disque",
    "sw_SKIP_LBL_COMPILE": "garder le rdb sur disque",
    "sw_REPORT": "rapport",
    "sw_PLANETS": "planètes connues (orbites et candidates)",
    "sw_SMALL_RV_SHIFT": "faibles variations de vitesse (template en une passe)",
    "run_name": "Nom du run",
    "run_name_auto": "auto",
    "wrap": "Le wrap",
    "wrap_hint": "ce qui sera lancé, tel qu'il sera écrit dans DATA_DIR/gui/runs/<run>/wrap.py ; vous pouvez le modifier avant de le lancer",
    "check": "Refaire depuis les réglages",
    "save_wrap": "Enregistrer en wrap_<nom>.py",
    "saved": "enregistré : %s",
    "errors": "Impossible de lancer",
    "warnings": "À noter",
    "queue_title": "File d'attente",
    "queue_empty": "rien en file",
    "stop": "Arrêter",
    "cancel": "Retirer de la file",
    "log": "Journal",
    "log_top": "Haut du journal (il cesse de suivre le run)",
    "step_goto": "Voir où elle commence dans le journal",
    "log_bottom": "Bas du journal (il suit à nouveau le run)",
    "archive_log_top": "Haut du journal (il cesse de suivre la recherche)",
    "archive_log_bottom": "Bas du journal (il suit à nouveau la recherche)",
    "state_queued": "en file",
    "state_running": "en cours",
    "state_done": "terminé",
    "state_failed": "échoué",
    "state_stopped": "arrêté",
    "state_cancelled": "annulé",
    "runs_window": "Runs lancés depuis la fenêtre",
    "runs_reports": "Rapports",
    "refresh": "Relire",
    "open": "ouvrir",
    "col_name": "nom",
    "col_state": "état",
    "col_started": "début",
    "col_duration": "durée",
    "col_reports": "rapports",
    "col_object": "objet",
    "col_report": "rapport",
    "col_date": "date",
    "nothing": "rien pour l'instant",
    "measure": "Mesurer à nouveau",
    "busy": "Un run tourne dans ce dossier : rien ne peut être supprimé avant qu'il se termine. %s",
    "measuring": "mesure en cours...",
    "kind_scratch": "jetable : refait par le prochain run qui en a besoin",
    "kind_rebuildable": "refaisable : refait en relançant",
    "kind_results": "résultats : gardés",
    "kind_data": "spectres : jamais touchés",
    "select_kind": "tout",
    "select_kind_hint": "cocher, ou décocher, tout ce qui est de cette catégorie",
    "select_type_hint": "cocher, ou décocher, tous ceux de ce type",
    "selected_summary": "%s cochés, %s",
    "type_cache": "cubes PCA2D",
    "type_superseded": "corrections remplacées",
    "type_corrected": "spectres corrigés",
    "type_cleaned": "spectres nettoyés",
    "type_lblrv": "vitesses par pose",
    "type_template": "templates",
    "type_mask": "masques",
    "delete_selected": "Supprimer la sélection",
    "confirm_delete": "Supprimer %s éléments, %s ? Un lien est supprimé en tant que lien ; ce vers quoi il pointe reste en place.",
    "deleted": "%s supprimés",
    "total": "total",
    "what_cache": "les cubes recalés de PCA2D, refaits par le prochain fit",
    "what_superseded": "une correction remplacée par une plus récente, et ce que LBL en a tiré",
    "what_spectra": "les spectres d'une cible",
    "what_corrected": "spectres corrigés (refaits à partir du fit ; à archiver d'abord s'ils sont destinés à quelqu'un)",
    "what_cleaned": "spectres nettoyés des telluriques",
    "what_lblrv": "vitesses de LBL par pose (une dizaine de minutes à refaire)",
    "what_template": "un template LBL",
    "what_mask": "un masque LBL",
    "what_rdb": "les tables de vitesses de LBL",
    "what_reports": "les rapports",
    "what_components": "bases de l'observateur et composantes des étoiles",
    "what_models": "les modèles de LBL : modèles stellaires, TAPAS, templates VETTED, bases PCA2D gardées",
    "lbl_says": "La description de LBL (en anglais) :",
    "help_data_dir": "Le dossier de données de LBL (DATA_DIR). Les spectres de chaque cible sont dans science/<objet>/ ; LBL écrit à côté ses templates, masques, vitesses par pose (lblrv), tables (lblrdb) et rapports.\n\n« Lire les cibles » lit une fois l'entête de chaque spectre (SNR, BERV, date, Teff) et garde le résultat : une relecture plus tard est rapide.",
    "help_instrument": "L'instrument LBL, et son mode, dont viennent les spectres. Il fixe la façon dont LBL lit les fichiers, le domaine de longueurs d'onde et la résolution. Deviné à partir des entêtes à la lecture des cibles ; NIRPS_HE et NIRPS_HA sont les deux modes de NIRPS.",
    "help_source": "Le pipeline qui a réduit les spectres (DATA_SOURCE : APERO, ESO, CADC...). Il fixe le format de fichier qu'attend LBL. Une base PCA2D faite à partir d'une autre source ne peut pas servir ici.",
    "help_python": "Le Python qui lance les wraps : celui de l'environnement conda où LBL (sa branche lbl-pca2d) est installé. Par défaut, le Python de cette fenêtre.",
    "help_pca2d_on": "Le pré-nettoyage : PCA2D sépare ce qui suit l'étoile de ce qui reste fixe sur le détecteur (résidus telluriques, airglow, défauts du détecteur), divise ces derniers des spectres, et LBL mesure ensuite les spectres corrigés, science/<objet>_pca2d/. Les spectres livrés ne sont jamais réécrits.\n\nDécoché, le wrap lance LBL seul, sur les spectres livrés.",
    "help_skip_pca2d": "Que faire d'une correction déjà sur disque (SKIP_LBL_PCA2D).\n\n« refaite seulement si la demande a changé » : gardée si les réglages, les fichiers d'entrée et le nominal.yaml de LBL sont ceux avec lesquels elle a été faite, refaite sinon. « la garder » : gardée même s'ils ont changé, ce qu'un avertissement signale. « la refaire » : refaite dans tous les cas ; la précédente est mise de côté, pas effacée.",
    "help_comparison": "Le template contre lequel LBL mesure chaque pose (OBJECT_COMPARISON).\n\n« le sien » : construit à partir des spectres de la cible. Le template d'une autre étoile (dans templates/, de type spectral voisin) convient à une cible qui a peu de spectres ou peu de couverture en BERV, dont le propre template serait bruité ou garderait des résidus telluriques. Avec PCA2D, c'est aussi le spectre de l'étoile dans le fit.",
    "help_teff": "La température effective de l'étoile (OBJECT_TEFF), en K. LBL s'en sert pour choisir le modèle PHOENIX derrière le masque de raies et la vitesse systémique. Lue dans les entêtes quand elles la donnent (en gris), sinon demandée à SIMBAD par le nom de la cible (la médiane des Teff qu'il a, sans celles loin du type spectral de l'étoile, sinon celle de son type spectral), dite sous le champ ; une valeur tapée ici la remplace.",
    "help_resproj": "Ajoute une colonne DTEMP3500 aux tables rdb : chaque pose projetée sur la dérivée d'un spectre à 3500 K par rapport à la température, lue comme un changement de température effective en K. Un indicateur d'activité (taches, facules) pour les naines M, à mettre à côté des vitesses.",
    "help_bounds": "Seuls les spectres entre ces dates entrent dans le fit PCA2D et sa correction. En RJD, BJD - 2 400 000, comme dans les tables de LBL (une borne entière tombe à midi UT, entre deux nuits aux Amériques). Vide : pas de borne. Les graphiques à gauche grisent ce que les bornes laissent de côté.",
    "help_run_name": "Le nom du run : son dossier, DATA_DIR/gui/runs/<nom>/, garde le wrap tel qu'il a tourné, tout son journal et run.json (quand, comment il s'est terminé, les rapports écrits). « auto » le construit à partir des cibles et des réglages.",
    "help_wrap": "Le wrap qui sera lancé : un script LBL ordinaire, qui se lance aussi tel quel à la main. Il suit ce qui est coché dans les autres onglets tant que vous ne le modifiez pas ici ; « Refaire depuis les réglages » le reconstruit. « Enregistrer » l'écrit dans le dossier de données sous wrap_<nom>.py.",
    "help_load": "Relire un wrap dans la fenêtre : il est analysé, jamais exécuté. Ses cibles, templates, étapes LBL et options PCA2D remplissent les onglets ; ce que la fenêtre n'affiche pas (INPUT_FILE, une clé PCA2D section.clé...) est gardé et réécrit dans chaque wrap qui en est tiré, de sorte qu'un wrap chargé puis réenregistré ne perd rien.",
    "help_sw_RUN_LBL_TELLUCLEAN": "La correction tellurique propre à LBL (un modèle TAPAS ajusté à chaque spectre), en deux passages : un premier à l'aveugle, un template fait de ses spectres nettoyés, puis un second sur les spectres livrés dont les raies de l'étoile sont divisées par ce template, pour qu'elles ne biaisent pas l'ajustement, et le template refait à partir d'eux. Écrite dans science/<objet>_tc/, que toutes les étapes suivantes utilisent. Pour les spectrographes dont le pipeline ne livre pas de correction tellurique (HARPS, ESPRESSO, ESPaDOnS...) ; désactivée pour les spectres NIRPS et SPIRou d'APERO, déjà corrigés. Avec PCA2D, le pré-nettoyage se fait sur les spectres nettoyés.",
    "help_sw_RUN_LBL_TEMPLATE": "Construire le template de l'étoile : tous ses spectres ramenés dans le référentiel de l'étoile et combinés en un spectre à haut SNR (templates/). Chaque raie est mesurée contre lui. Seulement pour une cible mesurée contre son propre template ; celui d'une autre étoile est lu tel quel.",
    "help_sw_RUN_LBL_MASK": "Construire la liste de raies (masks/) à partir du template : la position et le signe de chaque raie, les domaines sur lesquels les vitesses sont mesurées, et la vitesse systémique de l'étoile.",
    "help_sw_RUN_LBL_COMPUTE": "Mesurer la vitesse de chaque raie dans chaque pose contre le template (lblrv/, un fichier par pose). La plus longue des étapes de LBL, typiquement une dizaine de minutes.",
    "help_sw_RUN_LBL_COMPILE": "Combiner les vitesses par raie de chaque pose en une vitesse et son erreur, avec les indicateurs (d2v, d3v, DTEMP...), dans les tables rdb de lblrdb/ : lbl_ par pose, lbl2_ par nuit.",
    "help_sw_SKIP_LBL_TEMPLATE": "Garder un template déjà sur disque plutôt que de le refaire. À décocher quand les spectres ont changé (plus nombreux, une autre correction) : les vitesses doivent être mesurées contre un template fait des mêmes spectres. Avec PCA2D, une nouvelle correction met d'elle-même les anciens produits de côté.",
    "help_sw_SKIP_LBL_MASK": "Garder un masque déjà sur disque plutôt que de le refaire. À décocher quand le template a été refait.",
    "help_sw_SKIP_LBL_COMPUTE": "Ne pas remesurer une pose dont les vitesses par raie sont sur disque : seules les nouvelles poses sont mesurées. À décocher quand le template ou le masque a changé, sinon d'anciennes et de nouvelles vitesses se mélangent dans une même table.",
    "help_sw_SKIP_LBL_COMPILE": "Garder les tables rdb sur disque. Désactivé par défaut : compiler prend quelques secondes et doit inclure chaque nouvelle pose.",
    "help_sw_REPORT": "Écrire le rapport PDF dans reports/ : les vitesses, leurs périodogrammes, les indicateurs et, avec PCA2D, ses propres pages d'abord (les river plots, les composantes, les vitesses avant et après).",
    "help_sw_PLANETS": "Utiliser les planètes connues et candidates de l'étoile (les catalogues de planètes de LBL, et DATA_DIR/personal_planet_catalog.yaml) : le template est construit sur le BERV plus leur vitesse képlérienne, et le rapport ajuste leur K à leurs périodes. Les vitesses rapportées restent corrigées du seul BERV.",
    "help_sw_SMALL_RV_SHIFT": "Coché (par défaut) : la vitesse de l'étoile est prise constante, et son template est fait en une passe, chaque spectre décalé de son seul BERV. Juste quand la vitesse varie de moins d'environ 100 m/s sur les spectres (MAX_CONVERGENCE_TEMPLATE_RV) : planètes de quelques m/s à quelques dizaines de m/s, activité ; cela évite les passes de convergence du template. Décoché : la vitesse de chaque spectre est mesurée contre le template, et le template refait, jusqu'à 5 fois, jusqu'à ce que les décalages s'accordent à 100 m/s ; pour une binaire, une naine brune ou un Jupiter chaud (le K de HD 189733 est de 205 m/s), dont les raies seraient sinon étalées dans le template. Avec les planètes connues cochées et les orbites de l'étoile dans les catalogues, le template suit plutôt les orbites. Décoché, le référentiel de l'étoile de PCA2D suit les vitesses que le template a mesurées. Ce n'est pas le velocity_term de PCA2D, une vitesse par pose ajustée dans le fit PCA2D, qui ne change aucun template.",
    "launch": "Lancer",
    "launch_running": "En cours · %s",
    "launch_more": "Un run est en cours : un clic maintenant lance celui-ci à sa fin",
    "instrument_followed": "instrument réglé sur %s, celui des cibles cochées",
    "overwrite_q": "%s existe déjà : le remplacer ?",
    "err_no_folder": "le dossier de données %s n'existe pas",
    "err_no_instrument": "aucun instrument choisi",
    "err_no_target": "aucune cible cochée",
    "err_unknown_instrument": "LBL ne connaît pas l'instrument %s",
    "err_no_source": "%s n'a pas de source de données %s (LBL connaît %s)",
    "err_wrong_instrument": "%s contient des spectres %s, et l'instrument choisi est %s : choisissez l'instrument de ses spectres, ou décochez-la",
    "warn_no_teff": "pas de température effective pour %s : LBL reçoit %s K (elle ne sert qu'à choisir le modèle stellaire du masque)",
    "simbad_ask": "Redemander les Teff à SIMBAD",
    "simbad_asking": "SIMBAD...",
    "simbad_failed": "SIMBAD n'a pas répondu",
    "simbad_unknown": "SIMBAD ne connaît pas ce nom",
    "simbad_none": "SIMBAD : %s, sans Teff",
    "simbad_teff": "SIMBAD : %s K (%s mesures)",
    "simbad_teff_some": "SIMBAD : %s K (%s mesures sur %s)",
    "simbad_why": "%s pour SIMBAD, de type spectral %s. La médiane des Teff qu'il a, sans celles à plus d'un cinquième de celle de son type spectral (Pecaut & Mamajek 2013).",
    "simbad_teff_spt": "SIMBAD : %s K, d'après son type spectral %s",
    "simbad_use": "la prendre",
    "err_no_comparison": "%s : le template %s n'est pas dans templates/ et ne peut pas être construit à partir de spectres dans science/",
    "err_pca2d_kind": "PCA2D %s : %s n'est pas de type %s",
    "err_no_basis": "aucune base de l'observateur %s dans les sous-dossiers de %s, ni dans %s",
    "err_basis_other": "la base %s a été faite avec %s %s, ce run est %s %s : LBL la refuse",
    "err_pca2d": "PCA2D : %s",
    "basis_dropped": "la base figée %s n'est pas de %s %s : retirée",
    "plot_vtot": "SNR en fonction de v_tot, la vitesse de l'étoile dans le référentiel de l'observateur",
    "plot_vtot_hint": "un point par spectre, en fonction de v_tot = v_sys - BERV (composées relativistement), là où les raies de l'étoile tombent par rapport aux telluriques ; la bande ombrée, |v_tot| < 4 km/s, est là où elles tombent dessus. Les cibles dont la vitesse systémique est connue (SIMBAD, sinon les entêtes) ; toutes les cibles sont dans les graphiques du BERV",
    "help_vtot": "Là où les raies de l'étoile tombent sur le détecteur par rapport aux raies telluriques : sa vitesse systémique moins le BERV, pour chaque spectre.\n\nLa vitesse systémique est celle de SIMBAD, sinon celle des entêtes (PP_RV d'APERO quand sa source est une référence, ou la vitesse de la cible d'ESO) ; une cible qui n'en a pas est absente de ce graphique, et toutes les cibles sont dans ceux du BERV. Une étoile dont le v_tot reste à quelques km/s de 0 a ses raies sur les mêmes telluriques toute l'année : les résidus telluriques bougent alors avec elle et les deux référentiels se séparent mal, le cas d'une base de l'observateur figée.",
    "no_vsys": "pas de vitesse systémique dans les entêtes des cibles cochées",
    "n_spectra": "%s spectres",
    "tellu_off": "désactivé pour NIRPS et SPIRou : leurs spectres arrivent corrigés des telluriques par APERO",
    "vetted_group": "Templates validés de LBL",
    "vetted_label": "%s (%s K, %s spectres)",
    "best_match": "apparier un template",
    "best_match_all": "Apparier un template pour toutes",
    "best_match_none": "%s : pas de template validé pour %s K (2900-4300 K seulement)",
    "err_vetted_unknown": "%s n'est pas un des templates validés de LBL pour %s",
    "help_vetted": "Les templates validés de LBL : pour chaque tranche de 100 K de 3000 à 4200 K et pour chaque instrument, le template de l'étoile qui a le plus de spectres parmi les rotateurs lents (une période de rotation publiée, ou des raies étroites) qui ne sont pas des étoiles jeunes. Il est téléchargé du dépôt de LBL la première fois qu'un run le nomme (VETTED_<étoile>).\n\n« apparier un template » prend, pour une cible entre 2900 et 4300 K, le template validé de la température la plus proche : une cible qui a peu de spectres ou peu de couverture en BERV est alors mesurée contre un template de centaines de spectres.",
    "steps_title": "Étapes",
    "steps_planned": "telles que le wrap les prévoit ; elles se cochent au fil du run",
    "steps_of": "étapes de %s",
    "help_steps": "Les étapes du run d'un bout à l'autre, telles que le wrap les prévoit : pour chaque cible, LBL sur les spectres livrés (quand PCA2D les mesure d'abord), le cube, le fit, les figures et la correction de PCA2D, puis LBL sur les spectres corrigés et le rapport. Chacune se coche quand le journal atteint la suivante ; celle en cours est encadrée, une étape en échec est en rouge.",
    "step_telluclean": "nettoyage tellurique",
    "step_telluclean_1": "1er nettoyage tellurique, à l'aveugle",
    "step_template_tc": "template des spectres nettoyés",
    "step_telluclean_2": "2e nettoyage tellurique, divisé par ce template",
    "step_template_tc2": "template du 2e nettoyage",
    "step_template": "template",
    "step_mask": "masque",
    "step_compute": "vitesses",
    "step_compile": "table rdb",
    "step_report": "rapport",
    "step_pca2d_cube": "cube PCA2D",
    "step_pca2d_fit": "fit PCA2D",
    "step_pca2d_figures": "figures PCA2D",
    "step_pca2d_correct": "correction PCA2D",
    "no_data_dir": "Choisissez un dossier de données (DATA_DIR) ci-dessus : le dossier qui contient science/<objet>/ pour chaque cible.",
    "no_science": "%s n'existe pas. Un dossier de données contient un dossier par cible dans science/ : science/<objet>/, avec les spectres de cette cible.",
    "empty_science": "%s ne contient aucun dossier de cible. Chaque cible est un dossier science/<objet>/ avec ses spectres.",
    "read_of": "entêtes lus : %s sur %s",
    "err_mixed_title": "Deux instruments cochés.",
    "err_mixed": "Un run mesure les spectres d'un seul instrument (%s) : décochez les cibles de l'un des deux.",
    "err_vetted_missing": "%s n'est pas encore sur le dépôt de LBL : mettez %s dans %s, ou attendez que le dépôt l'ait",
    "warn_vetted_unknown": "le dépôt de LBL ne répond pas : %s sera téléchargé quand le run en aura besoin, s'il le peut",
    "sub_local": "Mes cibles",
    "sub_archive": "Recherche d'archive",
    "archive_title": "Spectres des archives publiques",
    "archive_target": "Cibles (plusieurs : séparez-les par des virgules, p. ex. GJ 1, GJ 2)",
    "archive_instruments": "Instruments",
    "archive_search": "Chercher dans les archives",
    "archive_fetch": "Récupérer les cochés (%s)",
    "archive_fetch_none": "Cochez ce qu'il faut récupérer",
    "archive_star": "%s, Teff %s K",
    "archive_searching": "recherche",
    "archive_found": "%s spectres",
    "archive_failed": "pas de réponse",
    "archive_stopped": "arrêtée",
    "archive_skipped": "hors de portée",
    "archive_search_stop": "Arrêter la recherche",
    "archive_log": "Ce que font la recherche et le téléchargement",
    "archive_nothing": "les archives interrogées n'ont pas de spectres publics de %s que LBL lit",
    "archive_progress": "%s : %s spectres sur %s, %s non récupérés",
    "archive_left": "reste environ %s",
    "archive_took": "fini en %s",
    "archive_waiting": "attend les autres téléchargements de cette archive",
    "archive_done": "récupérés dans %s : les cibles sont dans Mes cibles",
    "archive_big": "%s à télécharger dans %s : continuer ?",
    "archive_unavailable": "la recherche d'archive ne peut pas servir ici : %s",
    "col_reader": "lecteur LBL",
    "col_archive": "archive",
    "col_have": "déjà là",
    "col_size": "à récupérer",
    "col_bytes": "taille",
    "col_folder": "dossier",
    "err_archive_source": "%s a été récupérée pour %s %s : choisissez cet instrument et cette source de données",
    "err_layout_source": "%s contient des spectres au format de la source de données %s, et celle choisie est %s : choisissez %s",
    "help_archive": "Les spectres publics que LBL lit, d'une étoile ou de plusieurs (séparez leurs noms par des virgules), dans les archives que la fenêtre connaît (lbl.gui.fetch, repris de koloa dans LBL) : les t.fits de SPIRou au CADC, NIRPS, HARPS et ESPRESSO à l'ESO, HARPS-N au TNG, SOPHIE à l'OHP. Chaque nom est résolu par SIMBAD (son nom, ses coordonnées et sa Teff s'affichent sous le champ), puis chaque étoile est demandée à chaque archive en même temps, deux requêtes à la fois par archive. Une archive dont le télescope ne voit jamais l'étoile (moins de 20 degrés au-dessus de son horizon au mieux) n'est pas interrogée, et l'étoile y est dite « hors de portée » ; une archive qui ne répond pas est signalée, et Arrêter la recherche cesse de l'attendre.\n\nLes lignes cochées sont récupérées dans DATA_DIR/science/<étoile>/ (les calibrations dans DATA_DIR/calib/), quatre fichiers à la fois, les instruments en même temps, sans rien déplacer ni écraser ; un dossier qui contient les spectres d'un autre lecteur en reçoit un à part. Chaque dossier garde sa provenance, et le wrap en tire la source des données et le motif des fichiers. Le journal sous les résultats dit ce que chaque archive a répondu et pourquoi un fichier n'a pas été récupéré.",
    "refresh_targets": "Rafraîchir",
    "lang_switch": "EN",
    "theme_light": "Clair",
    "theme_dark": "Sombre",
    "tile_targets": "Cibles cochées",
    "tile_folder": "sur %s dans le dossier, %s spectres",
    "tile_spectra": "Leurs spectres",
    "tile_nights": "Leurs nuits",
    "tile_snr": "SNR médian %s",
    "tile_berv": "Leur couverture en BERV",
    "tile_none": "rien de coché",
    "close": "fermer",
    "yes": "oui",
    "no": "non",
}

#: the known options: a label in each language, and the help shown by their
#:    (i), (EN, FR). Written from lbl/doc/PCA2D.md and the comments of
#:    lbl/pca2d/nominal.yaml, where the measurements behind each nominal value
#:    are; a blank line in a help is a new paragraph
OPTIONS = {
    "n_star": (
        ("star spectra", "spectres de l'étoile"),
        ("How many spectra describe the star, its template included.\n\n"
         "1 (the nominal): the star is its template alone, carried to each"
         " exposure's BERV and taken out with a coefficient of exactly 1,"
         " never adjusted in depth. 2 adds one component in the star's frame"
         " with a free amplitude per exposure, to look at the star's own"
         " variability; it is never divided out of the corrected files. 0 is"
         " refused.\n\n"
         "1 gave the better velocities wherever it was measured (TOI-2120,"
         " GJ 1, TOI-4552). The fit's folder is named s<n_star>-o<n_earth>.",
         "Le nombre de spectres qui décrivent l'étoile, son template"
         " compris.\n\n"
         "1 (le nominal) : l'étoile est son template seul, porté au BERV de"
         " chaque pose et retiré avec un coefficient d'exactement 1, sans"
         " ajustement de profondeur. 2 ajoute une composante dans le"
         " référentiel de l'étoile, avec une amplitude libre par pose, pour"
         " regarder la variabilité propre de l'étoile ; elle n'est jamais"
         " divisée des fichiers corrigés. 0 est refusé.\n\n"
         "1 a donné les meilleures vitesses partout où cela a été mesuré"
         " (TOI-2120, GJ 1, TOI-4552). Le dossier du fit est nommé"
         " s<n_star>-o<n_earth>.")),
    "n_earth": (
        ("observer components", "composantes de l'observateur"),
        ("How many components fixed in the observer's frame are fitted:"
         " telluric residuals, airglow, detector features, everything that"
         " stays put on the detector while the star moves with the BERV."
         " They are divided out of the spectra LBL then measures.\n\n"
         "More components follow more of the atmosphere, but also have more"
         " freedom to take part of the star when the BERV moves little. 3 is"
         " the nominal for a star fitted on its own; a basis learned on"
         " several stars carries more (7 for NIRPS_SNR45-90).\n\n"
         "With a frozen basis (below), leave it unset to take all of its"
         " components.",
         "Le nombre de composantes fixes dans le référentiel de"
         " l'observateur : résidus telluriques, airglow, défauts du"
         " détecteur, tout ce qui reste en place sur le détecteur pendant"
         " que l'étoile se déplace avec le BERV. Elles sont divisées des"
         " spectres que LBL mesure ensuite.\n\n"
         "Plus de composantes suivent mieux l'atmosphère, mais ont aussi plus"
         " de liberté pour absorber une partie de l'étoile quand le BERV"
         " varie peu. 3 est le nominal pour une étoile ajustée seule ; une"
         " base apprise sur plusieurs étoiles en porte davantage (7 pour"
         " NIRPS_SNR45-90).\n\n"
         "Avec une base figée (plus bas), laissez vide pour prendre toutes"
         " ses composantes.")),
    "joint": (
        ("joint fit", "fit joint"),
        ("All the ticked targets fitted together against ONE observer"
         " basis, each star with its own spectrum.\n\n"
         "Stars observed at different times of the year sweep different"
         " BERVs, so together they separate the observer's frame from the"
         " stars' better than any one of them: this is how a shared basis"
         " is made (give it a name under 'save the observer basis as')."
         " Needs two targets or more.",
         "Toutes les cibles cochées ajustées ensemble contre UNE base de"
         " l'observateur, chaque étoile avec son propre spectre.\n\n"
         "Des étoiles observées à des moments différents de l'année"
         " balaient des BERV différents : ensemble, elles séparent le"
         " référentiel de l'observateur de celui des étoiles mieux qu'une"
         " seule. C'est ainsi qu'on fabrique une base partagée (donnez-lui"
         " un nom sous « enregistrer la base de l'observateur sous »). Il"
         " faut au moins deux cibles.")),
    "velocity_term": (
        ("a velocity per exposure", "une vitesse par pose"),
        ("Fit one velocity per exposure beside the components, as the"
         " derivative of the star's spectrum carried to that exposure.\n\n"
         "The star is never exactly where the BERV alone puts it (a planet,"
         " activity), and the observer components, free in every exposure,"
         " could take that motion; dividing them out would then remove part"
         " of the very signal the run is after. The velocity term keeps it"
         " out of them. It is fitted, never divided out, and written to the"
         " corrected files' header (PCASTR_V). Off by default.\n\n"
         "Not SMALL_RV_SHIFT (LBL tab), which decides whether LBL's template"
         " measures each exposure's velocity: this one lives inside the"
         " PCA2D fit and changes no template.",
         "Ajuster une vitesse par pose à côté des composantes, sous la forme"
         " de la dérivée du spectre de l'étoile portée à cette pose.\n\n"
         "L'étoile n'est jamais exactement là où le BERV seul la place (une"
         " planète, l'activité), et les composantes de l'observateur, libres"
         " dans chaque pose, pourraient absorber ce mouvement ; les diviser"
         " retirerait alors une partie du signal même que l'on cherche. Le"
         " terme de vitesse l'en tient à l'écart. Il est ajusté, jamais"
         " divisé, et écrit dans l'entête des fichiers corrigés (PCASTR_V)."
         " Désactivé par défaut.\n\n"
         "Ce n'est pas SMALL_RV_SHIFT (onglet LBL), qui décide si le template"
         " de LBL mesure la vitesse de chaque pose : celui-ci vit dans le fit"
         " PCA2D et ne change aucun template.")),
    "iters": (
        ("sweeps at most", "itérations au plus"),
        ("The most sweeps the fit makes, alternating the star's part and"
         " the observer's part; it stops earlier once converged. Raise it"
         " only if the log says the fit did not converge.",
         "Le nombre maximal de passes du fit, qui alterne la partie étoile"
         " et la partie observateur ; il s'arrête plus tôt s'il a convergé."
         " À augmenter seulement si le journal dit que le fit n'a pas"
         " convergé.")),
    "width_kms": (
        ("high pass (km/s)", "passe-haut (km/s)"),
        ("Width of the high-pass filter (Savitzky-Golay, on the log of the"
         " flux) applied to every spectrum before the fit. It removes the"
         " continuum and the blaze, and keeps the lines.\n\n"
         "In km/s so that a line is filtered the same way whatever the grid"
         " step. 100 km/s is far wider than any line, which go through"
         " untouched; narrower filters more and starts eating into broad"
         " features.",
         "Largeur du filtre passe-haut (Savitzky-Golay, sur le logarithme du"
         " flux) appliqué à chaque spectre avant le fit. Il retire le"
         " continu et le blaze, et garde les raies.\n\n"
         "En km/s pour qu'une raie soit filtrée de la même façon quel que"
         " soit le pas de la grille. 100 km/s est bien plus large que toute"
         " raie, qui passent intactes ; plus étroit filtre davantage et"
         " commence à entamer les structures larges.")),
    "dv": (
        ("grid step (km/s)", "pas de la grille (km/s)"),
        ("Step of the fit's common wavelength grid, in km/s per sample. The"
         " grid is uniform in log wavelength, so a Doppler shift is a pure"
         " translation on it.\n\n"
         "0.5 km/s oversamples the detector (a SPIRou pixel is about"
         " 2.3 km/s): a coarser step takes less memory and fits faster, a"
         " finer one brings nothing. Changing it rebuilds the cubes.",
         "Pas de la grille commune en longueur d'onde du fit, en km/s par"
         " échantillon. La grille est uniforme en log de la longueur d'onde,"
         " donc un décalage Doppler y est une simple translation.\n\n"
         "0.5 km/s suréchantillonne le détecteur (un pixel de SPIRou fait"
         " environ 2.3 km/s) : un pas plus grossier prend moins de mémoire"
         " et va plus vite, un pas plus fin n'apporte rien. Le changer"
         " refait les cubes.")),
    "nightly_stack": (
        ("coadd each night", "coaddition par nuit"),
        ("Coadd each night (or each bundle of a night, below) into one row"
         " of the fit, instead of one row per exposure.\n\n"
         "It is a memory choice: the fit on single exposures has strictly"
         " more information, but NIRPS takes three or four exposures per"
         " visit, so coadding divides the cube's size by as much. On for"
         " NIRPS. Every exposure is still corrected on its own, with its own"
         " amplitudes.",
         "Coadditionner chaque nuit (ou chaque paquet d'une nuit, plus bas)"
         " en une ligne du fit, au lieu d'une ligne par pose.\n\n"
         "C'est un choix de mémoire : le fit sur les poses individuelles a"
         " strictement plus d'information, mais NIRPS prend trois ou quatre"
         " poses par visite, et la coaddition divise d'autant la taille du"
         " cube. Activé pour NIRPS. Chaque pose est quand même corrigée"
         " individuellement, avec ses propres amplitudes.")),
    "bundle_berv": (
        ("night bundles, BERV (km/s)", "paquets par nuit, BERV (km/s)"),
        ("With nightly coadds, each night is cut into bundles spanning at"
         " most this much BERV, each bundle one row of the fit.\n\n"
         "The coadd is made in the observer's frame, so it smears the star"
         " by the BERV range it holds. At 0.1 km/s an ordinary NIRPS visit"
         " (4 to 35 m/s of BERV) stays one row, while a long sequence, such"
         " as a five-hour transit night (600 m/s), is cut into several."
         " Unset: whole nights.",
         "Avec la coaddition par nuit, chaque nuit est coupée en paquets"
         " couvrant au plus cet intervalle de BERV, chaque paquet étant une"
         " ligne du fit.\n\n"
         "La coaddition se fait dans le référentiel de l'observateur : elle"
         " étale donc l'étoile sur l'intervalle de BERV qu'elle contient. À"
         " 0.1 km/s, une visite NIRPS ordinaire (4 à 35 m/s de BERV) reste"
         " une seule ligne, alors qu'une longue séquence, comme une nuit de"
         " transit de cinq heures (600 m/s), est coupée en plusieurs. Vide :"
         " nuits entières.")),
    "bundle_minutes": (
        ("night bundles, minutes", "paquets par nuit, minutes"),
        ("With nightly coadds, also cut each night into bundles of at most"
         " this many minutes: over a long sequence the sky changes with the"
         " airmass (water column, airglow). A bundle closes as soon as one"
         " of the limits is reached. Unset: no limit in time.",
         "Avec la coaddition par nuit, couper aussi chaque nuit en paquets"
         " d'au plus ce nombre de minutes : sur une longue séquence, le ciel"
         " change avec la masse d'air (colonne d'eau, airglow). Un paquet se"
         " ferme dès qu'une des limites est atteinte. Vide : pas de limite"
         " en temps.")),
    "bundle_max": (
        ("night bundles, exposures at most",
         "paquets par nuit, poses au plus"),
        ("With nightly coadds, also at most this many exposures per bundle."
         " Unset: no limit.",
         "Avec la coaddition par nuit, au plus ce nombre de poses par"
         " paquet. Vide : pas de limite.")),
    "input_name": (
        ("frozen observer basis", "base de l'observateur figée"),
        ("Take the observer components from another fit, frozen, instead of"
         " fitting them on this star: only this star's spectrum and the"
         " components' amplitudes are fitted.\n\n"
         "For a star whose BERV barely moves (a short season), which cannot"
         " separate the two frames by itself: the basis comes from stars"
         " whose BERV sweeps. A frozen basis is divided out in full, not"
         " shrunk by its significance in this star's data.\n\n"
         "Only the bases of the same instrument and data source can be"
         " picked.",
         "Prendre les composantes de l'observateur d'un autre fit, figées,"
         " au lieu de les ajuster sur cette étoile : seuls le spectre de"
         " l'étoile et les amplitudes des composantes sont ajustés.\n\n"
         "Pour une étoile dont le BERV varie peu (une saison courte), qui ne"
         " peut pas séparer seule les deux référentiels : la base vient"
         " d'étoiles dont le BERV balaie. Une base figée est divisée en"
         " entier, sans rétrécissement selon sa significativité dans les"
         " données de cette étoile.\n\n"
         "Seules les bases du même instrument et de la même source de"
         " données sont proposées.")),
    "output_name": (
        ("save the observer basis as",
         "enregistrer la base de l'observateur sous"),
        ("Save this fit's observer components under this name"
         " (pca2d/components/PCA2D_Observer_<name>_<instrument>.fits), for"
         " other stars to take frozen. Unset: the object's name, or the"
         " joint fit's objects.",
         "Enregistrer les composantes de l'observateur de ce fit sous ce nom"
         " (pca2d/components/PCA2D_Observer_<nom>_<instrument>.fits), pour"
         " que d'autres étoiles les prennent figées. Vide : le nom de"
         " l'objet, ou les objets du fit joint.")),
    "name": (
        ("variant name", "nom de la variante"),
        ("A suffix that keeps variants apart: the corrected spectra go to"
         " science/<object>_pca2d_<name>/ and every LBL product follows that"
         " name, so several corrections of one star sit side by side and"
         " can be compared. Unset: <object>_pca2d.",
         "Un suffixe qui sépare les variantes : les spectres corrigés vont"
         " dans science/<objet>_pca2d_<nom>/ et tous les produits LBL"
         " suivent ce nom, de sorte que plusieurs corrections d'une même"
         " étoile coexistent et se comparent. Vide : <objet>_pca2d.")),
    "shrink": (
        ("divide only what is significant", "ne diviser que le significatif"),
        ("Divide each observer component out only where it is significant"
         " at that wavelength, all exposures together: its model is scaled"
         " by max(0, 1 - 1/z²), z its significance there. Where a component"
         " is noise, the file keeps its delivered flux rather than receive"
         " that noise.\n\n"
         "Does not apply to a frozen basis, divided out in full: this star's"
         " data cannot judge a basis learned on other stars.",
         "Ne diviser chaque composante de l'observateur que là où elle est"
         " significative à cette longueur d'onde, toutes les poses"
         " ensemble : son modèle est multiplié par max(0, 1 - 1/z²), z étant"
         " sa significativité à cet endroit. Là où une composante n'est que"
         " du bruit, le fichier garde son flux livré plutôt que de recevoir"
         " ce bruit.\n\n"
         "Ne s'applique pas à une base figée, divisée en entier : les"
         " données de cette étoile ne peuvent pas juger une base apprise sur"
         " d'autres étoiles.")),
    "weight": (
        ("correction metric (flux or the star's derivative)",
         "métrique de la correction (flux ou dérivée de l'étoile)"),
        ("How each sample is weighted when the components' amplitudes are"
         " measured again in every exposure, at the correction.\n\n"
         "velocity (the nominal): each sample weighted by (dT/dv)², the"
         " square of the derivative of the star's template there. A"
         " contaminant moves a velocity only through its overlap with that"
         " derivative, and where the star is flat it moves no line, so the"
         " amplitude is measured where it matters for the velocity. A floor"
         " (5% of the mean) keeps a component living where the star is flat"
         " from having no amplitude at all.\n\n"
         "flux: each sample weighted as in the fit (photon noise and"
         " telluric weights), the older behaviour.\n\n"
         "Only the measurement of the amplitudes changes: what is divided"
         " out, the blanked samples and the shrinkage do not. The two were"
         " indistinguishable on TOI-2120 and TOI-4552; velocity is nominal"
         " on principle.",
         "La pondération des échantillons quand les amplitudes des"
         " composantes sont mesurées à nouveau dans chaque pose, au moment"
         " de la correction.\n\n"
         "velocity (le nominal) : chaque échantillon pèse (dT/dv)², le carré"
         " de la dérivée du template de l'étoile à cet endroit. Un"
         " contaminant ne déplace une vitesse que par son recouvrement avec"
         " cette dérivée, et là où l'étoile est plate il ne déplace aucune"
         " raie : l'amplitude est donc mesurée là où elle compte pour la"
         " vitesse. Un plancher (5 % de la moyenne) évite qu'une composante"
         " qui ne vit que là où l'étoile est plate n'ait aucune amplitude.\n\n"
         "flux : chaque échantillon pèse comme dans le fit (bruit de photons"
         " et poids telluriques), l'ancien comportement.\n\n"
         "Seule la mesure des amplitudes change : ce qui est divisé, les"
         " échantillons masqués et le rétrécissement restent identiques. Les"
         " deux sont indiscernables sur TOI-2120 et TOI-4552 ; velocity est"
         " le nominal par principe.")),
    "mask": (
        ("samples blanked in the corrected files",
         "échantillons masqués dans les fichiers corrigés"),
        ("Which samples the corrected files blank (NaN), where the fit had"
         " nothing to say.\n\n"
         "common (the nominal): a sample is blanked in every exposure as"
         " soon as the fit gave it no weight in one of them (see the share"
         " below), so the whole campaign is measured on ONE set of lines."
         " exposure: each exposure blanks its own, so every epoch has a"
         " different set of lines; that alone cost 1.8 m/s of rms on"
         " TOI-4552. none: keep the delivered flux there.",
         "Quels échantillons les fichiers corrigés masquent (NaN), là où le"
         " fit n'avait rien à dire.\n\n"
         "common (le nominal) : un échantillon est masqué dans toutes les"
         " poses dès que le fit ne lui a donné aucun poids dans l'une"
         " d'elles (voir la fraction plus bas) ; toute la campagne est donc"
         " mesurée sur UN seul jeu de raies. exposure : chaque pose masque"
         " les siens, et chaque époque a un jeu de raies différent ; cela"
         " seul a coûté 1.8 m/s de rms sur TOI-4552. none : garder le flux"
         " livré à ces endroits.")),
    "mask_share": (
        ("share of exposures that must weight a sample",
         "fraction des poses qui doivent pondérer un échantillon"),
        ("With mask = common, the share of the exposures that must have"
         " given a sample weight for it to be kept.\n\n"
         "1 (the nominal) is unanimity: the strictest, and cheap when every"
         " exposure covers the same range. Below 1 the set of lines is"
         " decided by a majority, still one set for the whole campaign;"
         " useful when the exposures cover different ranges (on HD 10700,"
         " unanimity blanked 43% of the pixels once each exposure was"
         " shifted by its own BERV).",
         "Avec mask = common, la fraction des poses qui doivent avoir donné"
         " du poids à un échantillon pour qu'il soit gardé.\n\n"
         "1 (le nominal) est l'unanimité : la plus stricte, et peu coûteuse"
         " quand toutes les poses couvrent le même intervalle. Sous 1, le"
         " jeu de raies est décidé à la majorité, et reste unique pour toute"
         " la campagne ; utile quand les poses couvrent des intervalles"
         " différents (sur HD 10700, l'unanimité masquait 43 % des pixels"
         " une fois chaque pose décalée de son propre BERV).")),
    "nsig_cut": (
        ("residual clip (sigmas)", "coupure du résidu (sigmas)"),
        ("Blank every sample whose residual (the data less everything the"
         " fit models, panel 5 of the river plots) is beyond this many"
         " robust sigmas, measured locally (1.4826 MAD over the high pass's"
         " window): cosmic rays, bad pixels, local sky residuals.\n\n"
         "Together with the bias test below, on TOI-2120: rms from 17.0 to"
         " 14.6 m/s, without moving the planet's K. Unset: nothing clipped.",
         "Masquer tout échantillon dont le résidu (les données moins tout ce"
         " que le fit modélise, panneau 5 des river plots) dépasse ce nombre"
         " de sigmas robustes, mesurés localement (1.4826 MAD sur la fenêtre"
         " du passe-haut) : rayons cosmiques, mauvais pixels, résidus de"
         " ciel locaux.\n\n"
         "Avec le test de biais plus bas, sur TOI-2120 : rms de 17.0 à"
         " 14.6 m/s, sans déplacer le K de la planète. Vide : rien n'est"
         " coupé.")),
    "column_frac": (
        ("column clip: fraction", "colonnes : fraction coupée"),
        ("A column (one wavelength, in the observer's frame) where more than"
         " this fraction of the exposures was clipped by the residual cut,"
         " AND whose surviving exposures still have a reduced chi² above the"
         " threshold below, is blanked in every exposure: no exposure"
         " describes it, so it is a defect, a telluric line the correction"
         " does not reach or a sky residual.\n\n"
         "Pure noise never clips more than 2.2% of the exposures, so 10%"
         " cannot be reached by chance. Both conditions are needed.",
         "Une colonne (une longueur d'onde, dans le référentiel de"
         " l'observateur) où plus de cette fraction des poses a été coupée"
         " par le test du résidu, ET dont les poses restantes ont encore un"
         " chi² réduit au-dessus du seuil suivant, est masquée dans toutes"
         " les poses : aucune pose ne la décrit, c'est donc un défaut, une"
         " raie tellurique que la correction n'atteint pas ou un résidu de"
         " ciel.\n\n"
         "Le bruit pur ne coupe jamais plus de 2.2 % des poses : 10 % ne"
         " peut pas être atteint par hasard. Les deux conditions sont"
         " requises.")),
    "column_chi2": (
        ("column clip: chi2", "colonnes : chi2"),
        ("The second condition of the column test: the reduced chi² of the"
         " exposures left after clipping. Pure Gaussian noise gives 0.97,"
         " and never more than 1.27 over 4096 columns, so 1.5 is not reached"
         " by noise. Together, the two conditions caught 39 of 40 simulated"
         " bad columns and none of the 4056 good ones.",
         "La deuxième condition du test des colonnes : le chi² réduit des"
         " poses qui restent après la coupure. Du bruit gaussien pur donne"
         " 0.97, et jamais plus de 1.27 sur 4096 colonnes : 1.5 n'est pas"
         " atteint par le bruit. Ensemble, les deux conditions ont attrapé"
         " 39 des 40 mauvaises colonnes simulées et aucune des 4056"
         " bonnes.")),
    "bias_nsig": (
        ("observer-frame bias (sigmas)",
         "biais dans le référentiel de l'observateur (sigmas)"),
        ("A column whose MEAN residual over all the exposures (summed over"
         " two resolution elements) is beyond this many sigmas is blanked in"
         " every exposure.\n\n"
         "The residual cut compares each exposure to the others, so it"
         " cannot see a residual present in all of them; the mean over N"
         " exposures gains √N and shows whole lines left badly corrected"
         " (methane and water near 1761 to 1805 nm). On TOI-2120, 10 sigmas"
         " blanks about 10% of the domain; lower blanks more.",
         "Une colonne dont le résidu MOYEN sur toutes les poses (sommé sur"
         " deux éléments de résolution) dépasse ce nombre de sigmas est"
         " masquée dans toutes les poses.\n\n"
         "La coupure du résidu compare chaque pose aux autres : elle ne voit"
         " pas un résidu présent dans toutes. La moyenne sur N poses gagne"
         " √N et révèle des raies entières mal corrigées (méthane et eau"
         " vers 1761 à 1805 nm). Sur TOI-2120, 10 sigmas masquent environ"
         " 10 % du domaine ; plus bas masque davantage.")),
    "star_column_frac": (
        ("star frame: fraction clipped",
         "référentiel de l'étoile : fraction coupée"),
        ("The column test above, in the star's frame: each exposure carried"
         " by its own shift, a star column where more than this fraction of"
         " the exposures was clipped (and whose survivors are still above"
         " the chi² below) is blanked wherever it falls, in every exposure."
         " It catches what stands still on the star, as a narrow BERV range"
         " leaves it. Measured against another star's template, the static"
         " difference between the two stars (their metallicity) is expected"
         " and is taken out before this test. Off (unset) until measured on"
         " more stars than KIC 7847566.",
         "Le test des colonnes ci-dessus, dans le référentiel de l'étoile :"
         " chaque pose portée par son propre décalage, une colonne de"
         " l'étoile où plus de cette fraction des poses a été coupée (et"
         " dont les poses restantes dépassent encore le chi² suivant) est"
         " masquée là où elle tombe, dans toutes les poses. Il attrape ce"
         " qui reste immobile sur l'étoile, comme le laisse un intervalle de"
         " BERV étroit. Mesurée contre le template d'une autre étoile,"
         " la différence statique entre les deux étoiles (leur métallicité)"
         " est attendue et retirée avant ce test. Désactivé (vide) tant"
         " qu'il n'a pas été mesuré sur d'autres étoiles que KIC 7847566.")),
    "star_column_chi2": (
        ("star frame: chi2", "référentiel de l'étoile : chi2"),
        ("The second condition of the star-frame column test: the reduced"
         " chi² of the exposures left after clipping, as for the"
         " observer's frame. Off (unset) by default.",
         "La deuxième condition du test des colonnes dans le référentiel de"
         " l'étoile : le chi² réduit des poses qui restent après la coupure,"
         " comme dans le référentiel de l'observateur. Désactivé (vide) par"
         " défaut.")),
    "star_bias_nsig": (
        ("star frame: bias (sigmas)",
         "référentiel de l'étoile : biais (sigmas)"),
        ("The bias test, in the star's frame: a star column whose mean"
         " residual over the exposures is beyond this many sigmas is blanked"
         " in every exposure where it falls. Off (unset) by default.",
         "Le test de biais, dans le référentiel de l'étoile : une colonne de"
         " l'étoile dont le résidu moyen sur les poses dépasse ce nombre de"
         " sigmas est masquée dans toutes les poses où elle tombe. Désactivé"
         " (vide) par défaut.")),
    "min_pixel_snr": (
        ("least pixel SNR", "SNR minimal par pixel"),
        ("A sample whose own SNR (1/σ of the log flux, from the photon"
         " noise) is below this takes no part in the fit.\n\n"
         "A threshold in SNR, not in depth: the bottom of a deep line"
         " carries fewer photons but the most velocity information, and"
         " must stay. 5 is low on purpose: it drops dead pixels, it does not"
         " sort by brightness.",
         "Un échantillon dont le SNR propre (1/σ du log du flux, d'après le"
         " bruit de photons) est sous ce seuil ne participe pas au fit.\n\n"
         "Un seuil en SNR, pas en profondeur : le fond d'une raie profonde"
         " porte moins de photons mais le plus d'information en vitesse, et"
         " doit rester. 5 est bas exprès : il écarte les pixels morts, il ne"
         " trie pas par brillance.")),
    "compare": (
        ("measure the delivered spectra too",
         "mesurer aussi les spectres livrés"),
        ("LBL first measures the delivered spectra (science/<object>) too,"
         " so that the report puts the velocities before and after the"
         " correction side by side, exposure by exposure. Costs one more LBL"
         " run.",
         "LBL mesure d'abord aussi les spectres livrés (science/<objet>),"
         " pour que le rapport mette côte à côte les vitesses avant et après"
         " la correction, pose par pose. Coûte une passe de LBL de plus.")),
    "figures": (
        ("figures of the correction", "figures de la correction"),
        ("Draw the river plots and the other figures of the correction (the"
         " components, their amplitudes in time and against BERV and"
         " airmass, their periodograms) into the PCA2D report. Off makes the"
         " run faster and the report thinner.",
         "Dessiner les river plots et les autres figures de la correction"
         " (les composantes, leurs amplitudes dans le temps et en fonction"
         " du BERV et de la masse d'air, leurs périodogrammes) dans le"
         " rapport PCA2D. Désactivé, le run est plus rapide et le rapport"
         " plus mince.")),
    "overwrite": (
        ("redo even if on disk", "refaire même si sur disque"),
        ("Make the correction again even when the same one (same settings,"
         " same input files) is on disk. The previous corrected spectra and"
         " every LBL product made from them are moved to"
         " pca2d/<object>/superseded/<date>/; nothing is deleted. The same"
         " as 'redo it' under 'correction on disk'.",
         "Refaire la correction même quand la même (mêmes réglages, mêmes"
         " fichiers d'entrée) est sur disque. Les spectres corrigés"
         " précédents et tous les produits LBL qui en viennent sont déplacés"
         " dans pca2d/<objet>/superseded/<date>/ ; rien n'est effacé. Comme"
         " « la refaire » sous « correction sur disque ».")),
}


def strings(lang: str) -> dict:
    """The window's words in one language (English for an unknown one)."""
    return dict(FR if str(lang).lower().startswith("fr") else EN)


def option_text(key: str, lang: str, lbl_description: str = "") -> dict:
    """An option's label and help in one language: dict(label, help, lbl).
    `lbl` is LBL's own description (English), given only for an option the
    window has no help of its own for (one LBL added since)."""
    index = 1 if str(lang).lower().startswith("fr") else 0
    known = OPTIONS.get(key)
    label = known[0][index] if known else key
    helps = known[1] if known else None
    return dict(label=label, help=helps[index] if helps else "",
                lbl="" if helps else (lbl_description or ""))
