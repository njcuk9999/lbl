"""The window's words, in English and in French, under the same keys
(maintenance/tests/test_gui_app_i18n.py checks that neither language
lacks one).

OPTIONS below gives each PCA2D option a label and a help in both languages,
written from LBL's doc and the comments of its nominal.yaml. An option LBL
adds later is shown under its key, with LBL's own description (in English)
until it gets one here. The other (i) of the window are the help_* words.
"""

EN = {
    "title": "LBL",
    "banner_name": "line-by-line velocities",
    "tagline": "with or without the PCA2D precleaning: pick the targets, set, run, follow",
    "home": "LBL on GitHub",
    "eyebrow": "LBL · PCA2D",
    "intro": "The window writes an ordinary LBL wrap for the targets and options picked, runs it in the environment LBL is installed in, one run at a time, and shows its log as it goes. Every run keeps its wrap and its log in DATA_DIR/gui/runs.",
    "data_dir": "Data root",
    "data_dir_hint": "the folder holding Downloads/ and one LBL data folder per instrument and source (SPIROU_APERO/, NIRPS_HE_ESO/...)",
    "scan": "Read the targets",
    "scanning": "reading headers: %s of %s spectra",
    "scan_done": "%s targets, %s spectra",
    "data_folder": "Instrument folder (DATA_DIR)",
    "help_data_folder": "The LBL data folder the window works in: one of the root's, named after the instrument and data source whose spectra it holds. Choosing one sets the instrument and the data source of the wrap; the targets, the runs, the Data Explorer and the cleanup are those of this folder.",
    "folder_this": "science/ of the root (old layout) · %s targets, %s spectra",
    "folder_row": "%s · %s targets, %s spectra",
    "folder_none": "none yet: put spectra in Downloads, then Dispatch",
    "no_folder": "No instrument folder in %s yet: put the spectra you download in its Downloads/ folder, then Dispatch.",
    "downloads_n": "Downloads: %s files to dispatch",
    "downloads_title": "The files waiting in %s (those put aside in unused/ not counted); click to count them again",
    "dispatch": "Dispatch",
    "help_dispatch": "Sorts the files of Downloads/ into the instrument folders, from their headers alone:\n\n- the instrument and data source that read each one (its INSTRUME, NIRPS's HE or HA, the layout of its extensions);\n- what it is: a spectrum LBL reads (into science/<object>/: APERO's t.fits, ESO's S2D_A, for NIRPS at ESO its S2D_BLAZE_TELL_CORR_A, the DRS's e2ds_A...), a calibration (a blaze, into calib/, under the name the spectra give it), or another product of its pipeline (CCF, S1D...), put aside in Downloads/unused/;\n- its star: the files of one star go to one folder whatever their observers called it (HR1084, Eps_eri and K3_01 are all eps Eri), matched by where the star is in J2000, from the target's coordinates in the headers brought back to 2000 by their proper motion. The two stars of a binary stay apart: by their names (GJ 15 A, GJ 15 B), else by SIMBAD, which is asked the names of new stars and of places that carry several names.\n\nA star that already has a folder in the root keeps it; a new one is named as Naming of new stars says, and its folder can be given another name in the plan before anything moves. HARPS's EGGS, SOPHIE's HE and ESPRESSO's settings other than HR 1x1 get folders of their own (<object>_EGGS). On the way, gzipped spectra are gunzipped, SOPHIE's blaze is split out of its spectra, and the HARPS and HARPS-N spectra whose blaze ESO does not give get one made from them.\n\nNothing moves before you have seen the plan and clicked Move the files. Nothing is overwritten: a copy of a file already in place is taken out of Downloads, a different file of the same name is left there. Every move is written in Downloads/.dispatched.jsonl.\n\nThe models of every instrument folder (PHOENIX spectra, TAPAS, vetted templates...) are the root's: its models/ is a link to ROOT/models/, so that LBL downloads them once for every instrument (a models/ of its own is merged into the root's first, unless a file of it differs from the root's).",
    "dispatch_headers": "reading the headers: %s of %s",
    "dispatch_looking": "looking at the folders already in the root",
    "dispatch_simbad": "asking SIMBAD the stars' names: %s of %s",
    "dispatch_moving": "moving: %s of %s",
    "dispatch_blazes": "the files are in place; making the blazes ESO does not give, one a night: %s of %s",
    "dispatch_aside": "putting aside what LBL does not read: %s of %s",
    "dispatch_tidying": "removing the folders of Downloads left empty",
    "dispatch_models": "linking the models of the instrument folders to the root's",
    "dispatch_plan": "%s files in Downloads: %s spectra into %s folders, %s calibrations, %s put aside, %s left",
    "dispatch_go": "Move the files",
    "dispatch_cancel": "Cancel",
    "dispatch_new": "new",
    "dispatch_existing": "already there",
    "col_dispatch_folder": "Into",
    "col_dispatch_files": "Spectra",
    "col_dispatch_names": "Names in the headers",
    "dispatch_calib": "%s calibrations into %s/calib/",
    "dispatch_unused": "%s put aside in Downloads/unused/: %s (e.g. %s)",
    "dispatch_left": "%s left in Downloads: %s (e.g. %s)",
    "dispatch_either_star": "Files named %s could be either %s, where they go (the nearer), or %s: two stars SIMBAD knows apart",
    "dispatch_simbad_unreachable": "SIMBAD did not answer: the stars are told apart by their places and names only.",
    "dispatch_simbad_cut": "SIMBAD was asked %s names at most: the others are matched by their places only.",
    "dispatch_nothing": "Nothing waits in Downloads.",
    "dispatch_done": "%s files moved into their folders, %s copies of files already there taken out of Downloads, %s put aside in Downloads/unused/.",
    "dispatch_made": "Blazes made from the spectra (ESO does not give them): %s",
    "dispatch_kept": "Left in Downloads, a different file of that name being in its place: %s",
    "dispatch_failed": "Not moved: %s",
    "dispatch_error": "The dispatch stopped: %s",
    "dispatch_share_link": "models/ of %s: each made a link to the root's models/, one copy of LBL's models for every instrument",
    "dispatch_share_merge": "models/ of %s: their files moved into the root's models/ (copies of files already there removed), then each made a link to it",
    "dispatch_shared": "models/ of %s: links to the root's models/",
    "dispatch_share_kept": "%s/models kept as a folder of its own: %s differ from the root's",
    "dispatch_close": "Close",
    "tree_left": "%s left, which cannot be dispatched: %s (e.g. %s)",
    "pick": "Add files and dispatch…",
    "help_pick": "Spectra from anywhere on this machine, picked in the system's own window (several at once: Shift or Cmd, or Cmd-A for all the files of a folder). They are copied into a folder of Downloads of their own (moved, if ticked; a file already there byte for byte is not copied again), then their headers are read and the plan shows where each goes, as Dispatch does: nothing goes into the instrument folders before you have seen the plan and clicked Move the files.",
    "pick_prompt": "Spectra to add to the root",
    "pick_waiting": "choose the files in the system's window…",
    "pick_failed": "the system's window could not open: %s",
    "pick_summary": "%s files chosen (%s), from %s",
    "pick_move": "move them rather than copy them (they leave the folder they are in)",
    "pick_go": "Add and dispatch",
    "pick_added": "%s files added to %s (%s already there)",
    "dispatch_adding": "copying into Downloads: %s of %s",
    "dispatch_adding_move": "moving into Downloads: %s of %s",
    "tab_folders": "Folders",
    "tree_title": "What the root holds",
    "root_shown": "root: %s",
    "explore_planets": "published planets: %s",
    "setup_folder": "%s, from the instrument folder %s (the menu at the top)",
    "tree_old": "the root itself, old layout",
    "tree_old_note": "one data folder for every instrument; its tables and products stay readable (Data Explorer, Cleanup)",
    "tree_legacy_empty": "%s empty folders, no spectra",
    "tree_old_folders": "its LBL folders: %s",
    "template_lines": "Lines of interest",
    "template_lines_title": "The lines whose activity LBL measures (lbl/resources/activity_lines.yaml), at their vacuum wavelengths, shaded over the window LBL measures them in",
    "template_rest": "In the star's rest frame",
    "template_rest_title": "The template brought to its star's rest frame (its wavelengths divided by the Doppler factor of v_sys), where a species' lines are at their laboratory wavelengths; unticked, the template as LBL made it and the lines moved to it",
    "template_vsys": "v_sys (km/s)",
    "template_vsys_title": "The star's systemic velocity: the SYSTVELO of the mask LBL made from the template (measured against a PHOENIX model), else SIMBAD's; type another to try it, empty it to go back",
    "template_shifted": "brought to the star's rest frame, v_sys %s km/s (%s)",
    "template_vsys_typed": "typed",
    "template_vsys_unknown": "no systemic velocity known (no mask, none in SIMBAD): type one to bring the template to its star's rest frame",
    "template_lines_moved": "lines of interest moved to the template's frame by v_sys %s km/s",
    "folder_empty": "%s · no spectra yet",
    "tree_summary": "In the root: %s files to dispatch in Downloads · %s instrument folders, %s targets, %s spectra",
    "tree_downloads": "%s files to dispatch (%s), by name:",
    "tree_downloads_none": "nothing to dispatch",
    "tree_unused": "%s files put aside, not read by LBL",
    "tree_home": "%s targets, %s spectra",
    "tree_home_empty": "no spectra yet",
    "tree_open": "open",
    "naming": "Naming of new stars",
    "naming_apero": "APERO's database, else SIMBAD",
    "naming_simbad": "SIMBAD's main name",
    "naming_headers": "the headers' name (the most given)",
    "naming_hint": "a star already in the root keeps its folder's name: rename it in the tree below (✎)",
    "help_naming": "The name of the folder a star new to the root gets when it is dispatched:\n\n- APERO's database, else SIMBAD: the name APERO's astrometric database gives the star (GL699, EPSILONERI, TOI206, AN_SEX...), found by one of the names in the files' headers or by where the star is, so that a HARPS file named Gl699 or a programme's nickname (YO17) gets APERO's name too; APERO's files carry it already (DRSOBJN). The database is copied once from APERO's server (a few MB, kept in ~/.cache/lbl/gui). A star APERO does not know is named as SIMBAD does.\n- SIMBAD's main name: SIMBAD's main identifier of the star, without its prefixes (* eps Eri gives eps_Eri, NAME Barnard's star gives Barnard_s_star), found by a name of the headers that SIMBAD places there, else by asking SIMBAD for the brightest star where the files point.\n- The headers' name: the name the files' headers give the target most often (the observer's OBJECT, before APERO's DRSOBJN), made a folder's name.\n\nWhen none says a name, the next way does, and the headers last. A star that already has a folder in the root keeps that name, in every instrument folder (one star, one name): rename it in the tree. In the plan, the name of every new folder can be typed by hand before anything moves.",
    "col_dispatch_how": "Name from",
    "how_root": "the root",
    "how_existing": "its folder",
    "how_apero": "APERO",
    "how_simbad": "SIMBAD",
    "how_headers": "headers",
    "how_hand": "by hand",
    "dispatch_apero": "reading APERO's database of stars",
    "dispatch_apero_unreachable": "APERO's database of stars could not be fetched (its server did not answer): the new stars are named as SIMBAD names them.",
    "name_edit_title": "The folder's name: letters, digits, + - and _ (Enter to keep it, Esc to cancel)",
    "name_empty": "a name is needed",
    "name_space_ends": "no space at the start or the end of a name",
    "name_chars": "not in a folder's name: %s (letters, digits, + - and _ only; %s?)",
    "name_ends": "a letter or a digit first and last (%s?)",
    "name_long": "%s characters at most",
    "name_suffix": "%s ends the names of the folders LBL makes: another name",
    "name_same": "the same name",
    "name_taken": "%s is there already: another name",
    "rename_missing": "%s is not there (any more?)",
    "rename_busy": "a run works in the root (process %s): rename once it is done",
    "rename_nothing_new": "this star's files go to a folder already there: rename it in the tree",
    "rename_no_plan": "the plan is no longer shown: dispatch again",
    "tree_rename": "Rename this folder",
    "rename_ok": "Rename",
    "rename_everywhere": "also its folders of the same name in %s",
    "rename_will": "Renames %s to %s.",
    "rename_products": "%s: %s files and folders LBL made under the name %s keep it (e.g. %s): the next run makes them again under the new name, and the Cleanup tab can delete the old ones.",
    "rename_done": "%s → %s: %s folders renamed",
    "tree_current": "in use",
    "tree_empty": "%s empty folders, no targets: %s",
    "tree_calib": "%s files",
    "tree_models_link": "-> ../models, the root's",
    "tree_models_own": "a folder of its own (the next dispatch merges it into the root's)",
    "tree_models_none": "none yet (the next dispatch links it to the root's)",
    "tree_models": "shared by every instrument folder: %s files, %s",
    "tree_others": "Also in the root: %s",
    "targets_empty_hidden": "%s empty folders not shown (no spectra): %s",
    "dispatch_default_blaze": "%s: no default blaze yet, the first blaze of ESO's put there (models/default-blaze/), which LBL reads for the spectra whose night's blaze is missing",
    "dispatch_defaults": "Default blaze put in place: %s",
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
    "tab_explore": "Data Explorer",
    "explore_title": "Data explorer",
    "explore_rdb": "Star and flavour (lblrdb)",
    "explore_x": "x",
    "explore_y": "y",
    "explore_c": "Colour",
    "explore_nightly": "per night (lbl2)",
    "help_explore_nightly": "LBL writes two tables in each run: lbl_, one row per spectrum, and lbl2_, the spectra of each night combined into one row. The menus list the first; ticked, the plots and the FIP are those of the second, of the same run, for every table that has one. Greyed out when none has.",
    "help_explore": "The stars of the rdb tables LBL wrote in DATA_DIR/lblrdb, the newest first, and for each its flavours: what else the name of a table says, the PCA2D variant and the template (pca2d / GJ_1028_pca2d, VETTED_GL880...). One row per spectrum (the table of the nights, lbl2_, with the box beside the menus). Any numeric column against any other. The error bars are those of the column's error column when it has one (vrad: svrad, fwhm: sig_fwhm, DTEMP3500: sDTEMP3500); the colour is a third column, or each point's probability of being an outlier, read off the outlier-aware FIP of y (blue reliable, grey undecided, red outlier).\n\nDrag to zoom, double-click to see everything again; a point's file shows under the mouse. A new window opens on the newest table, its x the MJD, its y the velocity, its colour the BERV.\n\nOn the right, on the same y axis, the histogram of y (of each table as a fraction of its points when there are several). Under the plot, for each table, the rms of y (the standard deviation) and its robust rms (the median absolute deviation times 1.4826, the rms of a gaussian, which a few outliers do not move), as the report quotes them, and its median error bar.\n\n+ overplots another flavour of the same star (up to 8; another star once all are shown, or by its menu): the tables then take one colour each, the y of each is taken minus its median so that they sit on each other, and the menus offer the columns they all have. The x beside a table takes it off.",
    "explore_fip": "False inclusion probability",
    "help_explore_fip": "The false inclusion probability (Hara et al. 2022) of y, and of x when x is not a time, against the time of the table (rjd), over periods from 1.1 days to twice its span: the probability that a sinusoid at that period is NOT in the data, as -log10 (2 is 1 %, dotted; the higher, the surer).\n\nGaussian (dashed): every point taken at its error bar. Outlier-aware (solid): each point counts for its probability of being a normal one, so that a bad night neither invents a signal nor buries one; where the two part, the answer depends on how the outliers are treated. Each model fits its own jitter. They are those of the report (lbl/report/fip.py). A column without an error column is taken at its robust scatter. With several tables, the FIP of the y of each, in its colour (the gaussian one dashed).\n\nThe dash-dotted lines are the periods of the star's published planets, named: those of the catalogs LBL's PLANETS option reads (lbl/resources/planets: exoplanet.eu and the NASA Exoplanet Archive, after a personal_planet_catalog.yaml of yours), those without a complete orbit included.",
    "star_heading": "The star",
    "help_explore_star": "The star of each table shown, one card per star (its name folds it). From SIMBAD (at the CDS, else its mirrors at the CfA and IDIA): its names, type, spectral type, position, parallax, proper motion and systemic velocity, V G J H Ks, the medians of the Teff, [Fe/H], log g and v sin i it lists. From Gaia DR3 (ESA's archive, else GAVO's copy): BP, RP, parallax, proper motion, RUWE and, from ESA only, the GSP-Phot and FLAME parameters. From AllWISE (GAVO, else IRSA): W1 and W2 of the source nearest to the star taken back to 2010.5.\n\nDerived as the star-o-meter derives them. An M dwarf (M_G from 7.5 to 16.5, or type M): [Fe/H] from W1-W2, BP-RP and M_G (Duque-Arribas et al. 2023, Eq. 4; for -0.10 <= W1-W2 <= 0.24 and 1.85 <= G-J <= 3.10: a bright star's saturated W1 gives none), Teff from BP-RP with J-H or [Fe/H] (Mann et al. 2015; 2700 to 4100 K), R and M from M_Ks (Mann et al. 2015; 4.6 < M_Ks < 9.8; R with [Fe/H] times 1 + 0.04458 [Fe/H]), M from M_K (Delfosse et al. 2000; 4.5 to 9.5), log g and L from them. An AFGK dwarf: Gaia's GSP-Phot and FLAME, else the dwarf sequence of Pecaut & Mamajek (2013) at its BP-RP and M_G, or at its spectral type. Nothing for a giant, a white dwarf or an O or B star. A relation outside its range gives nothing, and says why.\n\nPublished: SIMBAD's medians, and the NASA Exoplanet Archive's parameters of the host, each with its paper. Planets: the archive's composite table (pscomppars), matched by the star's Gaia DR3, TIC, HD and HIP numbers and its names: period, K, mass (\"sin i\" when only that is known), radius, semi-major axis, eccentricity, equilibrium temperature, how and when found, the discovery and orbit papers. When the archive cannot be reached, the planets of LBL's catalog (period, K).\n\nThe tellurics: the days of the year when the star's velocity seen from the Earth's centre, RV - BERV (composed relativistically), is within 5 km/s: its lines then sit on the telluric lines. The BERV leaves out the Earth's rotation (under 0.5 km/s).\n\nAsked once, kept in DATA_DIR/gui/star_info.json for 30 days; Ask again asks them again.",
    "star_asking": "asking SIMBAD, Gaia, AllWISE and the NASA Exoplanet Archive...",
    "star_refresh": "Ask SIMBAD, Gaia, AllWISE and the NASA Exoplanet Archive again",
    "star_again": "Ask again",
    "star_unknown": "SIMBAD knows none of these names: %s",
    "star_failed": "Could not be asked: %s",
    "star_n_planets": "%s published planets",
    "star_one_planet": "1 published planet",
    "star_unreached": "not reached",
    "star_no_planet": "no published planet",
    "star_g_position": "Position and motion",
    "star_g_photometry": "Photometry",
    "star_g_derived": "Derived: %s",
    "star_g_published": "Published",
    "star_mode_mdwarf": "M dwarf",
    "star_mode_afgk_gaia": "AFGK dwarf, Gaia DR3",
    "star_mode_afgk_seq": "AFGK dwarf, dwarf sequence",
    "star_mode_wd": "white dwarf, none",
    "star_mode_other": "not a dwarf, none",
    "star_mode_early": "O or B star, none",
    "star_mode_none": "none",
    "star_published_none": "nothing listed",
    "star_median": "median of %s",
    "star_note_out": "%s = %s is outside %s to %s (%s): not derived.",
    "star_note_duque_missing": "[Fe/H]: W1, W2, BP, RP or the parallax missing.",
    "star_note_no_bprp": "Teff: no Gaia BP-RP.",
    "star_note_no_mk": "R and M: no Ks or no parallax.",
    "star_note_sequence": "On the dwarf sequence near %s.",
    "star_tellurics": "Lines on the tellurics (|RV - BERV| < %s km/s):",
    "star_tellurics_never": "never",
    "star_tellurics_range": "RV %s km/s, BERV from %s to %s km/s in %s",
    "star_tellurics_no_rv": "Lines on the tellurics: no systemic velocity known.",
    "star_window": "%s to %s",
    "star_planets": "Published planets (NASA Exoplanet Archive)",
    "star_planets_lbl": "Planets (LBL's catalog: the NASA Exoplanet Archive could not be reached)",
    "star_planets_none": "None in the NASA Exoplanet Archive.",
    "star_planets_none_lbl": "None in LBL's catalog (the NASA Exoplanet Archive could not be reached).",
    "star_pl_name": "planet",
    "star_pl_period": "P [d]",
    "star_pl_k": "K [m/s]",
    "star_pl_mass": "M [M⊕]",
    "star_pl_radius": "R [R⊕]",
    "star_pl_a": "a [au]",
    "star_pl_e": "e",
    "star_pl_teq": "Teq [K]",
    "star_pl_method": "found",
    "star_pl_refs": "papers",
    "star_pl_controversial": "flagged as controversial by the archive",
    "star_sources": "SIMBAD: %s · Gaia DR3: %s · AllWISE: %s · planets: %s · asked %s",
    "star_nasa_down": "LBL's catalog",
    "template_explorer": "Template explorer",
    "help_template_explorer": "The template the table was measured against (DATA_DIR/templates, or models/template_* for a vetted one) and, under it, a quarter as high and over the same wavelengths, the TAPAS transmission of each molecule of the atmosphere at airmass 1, from 1 (clear) to 0 (opaque).\n\nDrag over either one to zoom in wavelength (the flux follows), double-click for the whole template; the wheel zooms too. Zoomed in, every point of the template is drawn; zoomed out, the lowest and highest of each pixel of the screen (the lowest transmission for TAPAS). A click on a molecule in the legend hides it.\n\nThe TAPAS file is one in DATA_DIR/models/tellurics with a column per molecule (wavelength, H2O, O2, CH4, CO2, O3, N2O, NO2), the one named after the instrument's site if there are several (LaSilla, MaunaKea...), else LBL's tapas_lbl.fits, which has water and the other molecules together. TAPAS is in the observer's frame and the template in the barycentric one: on the template, each telluric line moves with the BERV of each spectrum.\n\nWith several tables, the template of each in its colour (one for the tables that share it), each divided by its median over the wavelengths shown; TAPAS is that of the first table's instrument.\n\nThe template is shown in its star's rest frame, moved by the systemic velocity of the mask LBL made from it (SYSTVELO, measured against a PHOENIX model; else SIMBAD's, or the one you type): the lines of interest (activity_lines.yaml, in vacuum) are shaded where LBL measures them. Unticked, the template is as LBL made it and the lines are moved to it. A box cuts x and y; a box drawn flat, a scroll in x or a pan keep y fitted to what is seen: twice the span of its 3rd to 97th percentiles about their middle, no wider than its extremes, so that a few spurious points do not set the scale; a double click shows all of it.",
    "explore_no_plotly": "The plots of this tab are drawn by Plotly, which the page fetches from cdn.jsdelivr.net and could not get: is this machine offline? The other tabs do without it.",
    "explore_no_rdb": "No rdb table in %s yet: they come out of LBL's compile step.",
    "explore_unreadable": "Could not be read: %s.",
    "explore_add": "+ overplot",
    "explore_hist_count": "N",
    "explore_hist_fraction": "fraction",
    "explore_col_table": "table",
    "explore_col_n": "points",
    "explore_col_rms": "rms",
    "explore_col_robust": "robust rms",
    "explore_col_sigma": "median error",
    "explore_remove": "take this table off the plots",
    "explore_minus_median": "%s minus its median",
    "explore_colour_tables": "one colour per table, the y of each minus its median",
    "explore_colour_none": "none",
    "explore_outlier": "P(outlier), outlier-aware FIP of y",
    "explore_outlier_short": "P(outlier)",
    "explore_rows": "%s rows",
    "explore_bars": "error bars of y: %s",
    "explore_no_bars": "%s has no error column: no error bars",
    "explore_outlier_wait": "the outlier probabilities come with the FIP",
    "explore_fip_no_time": "This table has no rjd column: no FIP.",
    "explore_fip_wait": "computing the FIP of %s...",
    "explore_fip_none": "fewer than 10 points with an error: no FIP",
    "explore_gauss": "gaussian",
    "explore_oa": "outlier-aware",
    "explore_fip_best": "%s: lowest outlier-aware FIP at %s d (%s); %s of %s points more likely outliers than not",
    "explore_fip_noerr": "no error column: taken at its robust scatter",
    "explore_year": "1 yr",
    "explore_period_axis": "Period (days)",
    "explore_fip_axis": "-log10 FIP (1 % dotted)",
    "template_none": "No template found for this table, in DATA_DIR/templates or models/template_*.",
    "template_flux": "Template flux",
    "template_flux_median": "Template flux / its median",
    "template_several": "templates: %s",
    "template_missing": "no template found for %s",
    "template_transmission": "Transmission",
    "template_transmission_short": "transmission",
    "template_others": "others",
    "template_wavelength": "Wavelength (nm)",
    "template_of": "template: %s",
    "template_tapas": "TAPAS: %s, airmass 1, observer's frame",
    "template_tapas_lbl": "TAPAS: %s (water, and the others together), airmass 1, observer's frame",
    "template_berv": "the telluric lines move over the template with the BERV, here from %s to %s km/s",
    "template_no_tapas": "no TAPAS file in DATA_DIR/models/tellurics",
    "tab_refs": "References",
    "tab_docs": "Documentation",
    "docs_search": "Search the documentation",
    "docs_toc": "On this page",
    "docs_english": "",
    "docs_prev": "← %s",
    "docs_next": "%s →",
    "docs_none": "Nothing found for %s.",
    "docs_file": "This page: %s",
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
    "sw_PLANETS": "known planets (their orbits)",
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
    "runs_window": "Runs from the window",
    "runs_reports": "Reports",
    "refresh": "Read again",
    "open": "open",
    "col_name": "name",
    "col_state": "state",
    "col_started": "started",
    "col_duration": "duration",
    "col_reports": "reports",
    "col_basis": "PCA2D basis",
    "freeze_basis": "Freeze basis",
    "freeze_title": "Keep this run's observer basis (%s) under a name, in models/pca2d_<INSTRUMENT>_<SOURCE>/: the later PCA2D runs of this instrument and source can use it frozen (input_name), and no cleanup removes it",
    "freeze_placeholder": "a name: nirps_2026a",
    "freeze_go": "Freeze",
    "freeze_will": "Kept as %s; the PCA2D settings of this instrument and source offer it as %s.",
    "freeze_done": "Basis frozen as %s: %s",
    "basis_frozen_as": "frozen as %s",
    "basis_used": "read the frozen basis %s",
    "basis_replaced": "replaced by a later fit of %s",
    "basis_missing": "no basis on disk",
    "basis_running": "once the run is over",
    "freeze_taken": "%s is the name of a basis already there (%s): another name",
    "freeze_target": "%s is a target of this folder: its next fit would write over the frozen basis; another name",
    "freeze_already": "this basis is already frozen, as %s",
    "freeze_no_source": "%s does not say its instrument and data source (INSTRUME, DATASRC): it cannot be filed",
    "freeze_state_used": "this run read a frozen basis: it fitted none of its own",
    "freeze_state_replaced": "a later fit wrote over this run's basis",
    "freeze_state_missing": "this run's basis is no longer on disk",
    "freeze_state_running": "the run is not over",
    "freeze_state_none": "this run fitted no PCA2D basis",
    "col_object": "object",
    "col_report": "report",
    "col_date": "date",
    "nothing": "nothing yet",
    "measure": "Measure again",
    "busy": "A run is going in this folder, so nothing can be deleted until it ends: %s",
    "busy_unknown": "Whether a run is going in this folder cannot be told on this system (neither psutil nor lsof is there), so nothing can be deleted: install psutil (pip install psutil).",
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
    "what_residuals": "PCA2D's residuals of each exposure (remade from the fit, with save_residuals)",
    "what_cleaned": "telluric-cleaned spectra",
    "what_lblrv": "LBL's per-exposure velocities (about ten minutes to remake)",
    "what_template": "an LBL template",
    "what_mask": "an LBL mask",
    "what_rdb": "LBL's velocity tables",
    "what_reports": "the reports",
    "what_components": "observer bases and stars' components",
    "what_models": "LBL's models: stellar models, TAPAS, vetted templates, kept PCA2D bases",
    "what_downloads": "downloaded, waiting to be dispatched",
    "what_unused": "products LBL does not read, put aside by the dispatch (CCF, S1D...)",
    "explore_root_home": "root, old layout",
    "lbl_says": "LBL's own description:",
    "help_data_dir": "The root of your LBL data. Put what you download in its Downloads/ folder, and Dispatch sorts it: each spectrum into ROOT/{INSTRUMENT}_{SOURCE}/science/{OBJECT}/ (ROOT/{INSTRUMENT}/ for an instrument LBL reads from one source only: ESPRESSO, SOPHIE...), with LBL's names of the instruments and data sources. Each of these folders is an LBL data folder (DATA_DIR): the wrap is given it, and LBL writes its templates, masks, per-exposure velocities (lblrv), tables (lblrdb) and reports in it.\n\nA folder that holds science/<object>/ itself (the old layout) still works: choose '(this folder)' in the menu below.\n\n'Read the targets' reads every spectrum's header once (SNR, BERV, date, Teff) and keeps the result, so reading again later is quick.",
    "help_instrument": "The LBL instrument, and its mode, the spectra come from. It sets how LBL reads the files, the wavelength domain and the resolution. Guessed from the headers when the targets are read; NIRPS_HE and NIRPS_HA are NIRPS's two modes.",
    "help_source": "The pipeline that reduced the spectra (DATA_SOURCE: APERO, ESO, CADC...). It sets the file format LBL expects. A PCA2D basis made from another source cannot be used here.",
    "help_python": "The Python that runs the wraps: that of the environment LBL is installed in (the installation steps explain it). By default, the Python this window runs with.",
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
    "help_sw_RUN_LBL_TEMPLATE": "Build the star's template: all its spectra shifted into the star's frame and combined into one high-SNR spectrum (templates/). No spectrum is resampled: every pixel of every spectrum keeps its own wavelength, the spectra together sample the star far more finely than the detector, and the template and its derivatives are a local polynomial fitted through them, over one resolution element (lbl/doc/TEMPLATE.md). Every line is measured against it. Only for a target measured against its own template; another star's is read as it is.",
    "help_sw_RUN_LBL_MASK": "Build the line list (masks/) from the template: each line's position and sign, the domains the velocities are measured over, and the star's systemic velocity.",
    "help_sw_RUN_LBL_COMPUTE": "Measure the velocity of every line in every exposure against the template (lblrv/, one file per exposure). The longest of LBL's steps, typically about ten minutes.",
    "help_sw_RUN_LBL_COMPILE": "Combine each exposure's per-line velocities into one velocity and its error, with the indicators (d2v, d3v, DTEMP...), into the rdb tables of lblrdb/: lbl_ per exposure, lbl2_ per night.",
    "help_sw_SKIP_LBL_TEMPLATE": "Keep a template already on disk rather than build it again. Untick it when the spectra changed (more of them, another correction): the velocities must be measured against a template made from the same spectra. With PCA2D, a new correction moves the old products aside by itself.",
    "help_sw_SKIP_LBL_MASK": "Keep a mask already on disk rather than build it again. Untick it when the template was made again.",
    "help_sw_SKIP_LBL_COMPUTE": "Do not measure again an exposure whose per-line velocities are on disk: only the new exposures are measured. Untick it when the template or the mask changed, or old and new velocities get mixed in one table.",
    "help_sw_SKIP_LBL_COMPILE": "Keep the rdb tables on disk. Off by default: compiling takes seconds and must include every new exposure.",
    "help_sw_REPORT": "Write the PDF report in reports/: the velocities, their periodograms, the indicators and, with PCA2D, its own pages first (the river plots, the components, the velocities before and after).",
    "help_sw_PLANETS": "Use the star's known planets: the published orbits of its confirmed planets (LBL's catalog, lbl/resources/planets/known_planets.yaml), or your own solutions (DATA_DIR/personal_planet_catalog.yaml), which replace them for that star. The template is built on the BERV plus their Keplerian velocity, and the report fits their K at their periods. The velocities reported stay corrected for the BERV only.",
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
    "err_basis_grid": "the basis %s is not on the grid of this run, IT MUST BE REBUILT (run again the joint fit that made it, with this LBL); LBL refuses it: %s",
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
    "help_steps": "The steps of the run from end to end, as the wrap plans them: for each target, LBL on the delivered spectra (when PCA2D measures them first), PCA2D's cube, fit, figures and correction, then LBL on the corrected spectra and the report. Each one ticks off when the log reaches the next; the one going is outlined, a failed one is red. Every step that runs has its pill: a template or a mask whose step is unticked but that is not on disk is made all the same before the velocities, and its pill is dashed (its tip says why); a step the log names that the plan did not have gets a dashed pill where it comes.",
    "step_telluclean": "telluric cleaning",
    "step_telluclean_1": "telluric cleaning 1, blind",
    "step_template_tc": "template of the cleaned spectra",
    "step_telluclean_2": "telluric cleaning 2, divided by that template",
    "step_template_tc2": "template of the second cleaning",
    "step_template": "template",
    "step_mask": "mask",
    "step_template_needed": "template",
    "step_mask_needed": "mask",
    "step_added": "Not in the plan: the log says the run made it (a template or a mask that was missing, made before the velocities)",
    "step_needed_tip": "Its step is unticked in the LBL settings, but it is not on disk and the velocities need it: the run makes it all the same",
    "step_compute": "velocities",
    "step_compile": "rdb table",
    "step_report": "report",
    "step_pca2d_cube": "PCA2D cube",
    "step_pca2d_fit": "PCA2D fit",
    "step_pca2d_figures": "PCA2D figures",
    "step_pca2d_correct": "PCA2D correction",
    "no_data_dir": "Choose a data root above: the folder holding Downloads/ and the instrument folders.",
    "no_science": "%s does not exist. A data folder holds one folder per target in science/: science/<object>/, with that target's spectra.",
    "empty_science": "%s holds no spectra: each target is a folder science/<object>/ with its spectra.",
    "read_of": "headers read: %s of %s",
    "err_mixed_title": "Two instruments ticked.",
    "err_mixed": "A run measures one instrument's spectra (%s): untick the targets of one of them.",
    "keep_family": "Keep %s, untick the others",
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
    "err_folder_source": "%s holds the spectra of %s %s: choose that instrument and data source, or the instrument folder of the one chosen",
    "err_layout_source": "%s holds spectra laid out for the data source %s, and the one chosen is %s: choose %s",
    "help_archive": "The public spectra that LBL reads, of one star or of several (separate their names with commas), in the archives the window knows: SPIRou's t.fits at the CADC, NIRPS, HARPS and ESPRESSO at ESO, HARPS-N at the TNG, SOPHIE at the OHP. Each name is resolved by SIMBAD (its name, its coordinates and its Teff are shown under the field), then every star is asked of every archive at once, two requests at a time in one archive. An archive whose telescope never sees the star (less than 20 degrees above its horizon at best) is not asked, said 'out of reach'; one that does not answer is said, and Stop the search stops waiting for it.\n\nThe ticked rows are fetched into the root's folder of their instrument and data source, {INSTRUMENT}_{SOURCE}/science/<star>/ (the folder the star already has there, if any; calibrations in its calib/), four files at a time, the instruments at once, nothing there moved or overwritten; a folder that holds another reader's spectra gets a folder of its own. Each folder keeps what it came from, and the wrap takes the data source and the file pattern from it. The log under the results says what each archive answered and why a file was not fetched.",
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
    "home": "LBL sur GitHub",
    "eyebrow": "LBL · PCA2D",
    "intro": "La fenêtre écrit un wrap LBL ordinaire pour les cibles et les options choisies, le lance dans l'environnement où LBL est installé, un run à la fois, et en affiche le journal au fil du calcul. Chaque run garde son wrap et son journal dans DATA_DIR/gui/runs.",
    "data_dir": "Racine des données",
    "data_dir_hint": "le dossier qui contient Downloads/ et un dossier de données LBL par instrument et source (SPIROU_APERO/, NIRPS_HE_ESO/...)",
    "scan": "Lire les cibles",
    "scanning": "lecture des en-têtes : %s spectres sur %s",
    "scan_done": "%s cibles, %s spectres",
    "data_folder": "Dossier d'instrument (DATA_DIR)",
    "help_data_folder": "Le dossier de données LBL dans lequel la fenêtre travaille : un de ceux de la racine, nommé d'après l'instrument et la source de données dont il contient les spectres. En choisir un fixe l'instrument et la source de données du wrap ; les cibles, les exécutions, l'explorateur de données et le ménage sont ceux de ce dossier.",
    "folder_this": "science/ de la racine (ancienne organisation) · %s cibles, %s spectres",
    "folder_row": "%s · %s cibles, %s spectres",
    "folder_none": "aucun pour l'instant : mettez des spectres dans Downloads, puis Répartir",
    "no_folder": "Aucun dossier d'instrument dans %s pour l'instant : mettez les spectres que vous téléchargez dans son dossier Downloads/, puis Répartir.",
    "downloads_n": "Downloads : %s fichiers à répartir",
    "downloads_title": "Les fichiers en attente dans %s (ceux mis de côté dans unused/ ne sont pas comptés) ; cliquez pour les recompter",
    "dispatch": "Répartir",
    "help_dispatch": "Classe les fichiers de Downloads/ dans les dossiers d'instrument, d'après leurs en-têtes seulement :\n\n- l'instrument et la source de données qui lisent chacun (son INSTRUME, HE ou HA pour NIRPS, la disposition de ses extensions) ;\n- ce qu'il est : un spectre que LBL lit (dans science/<objet>/ : les t.fits d'APERO, les S2D_A de l'ESO, pour NIRPS à l'ESO ses S2D_BLAZE_TELL_CORR_A, les e2ds_A du DRS...), une calibration (un blaze, dans calib/, sous le nom que lui donnent les spectres), ou un autre produit de son pipeline (CCF, S1D...), mis de côté dans Downloads/unused/ ;\n- son étoile : les fichiers d'une même étoile vont dans un même dossier, quel que soit le nom que leurs observateurs lui ont donné (HR1084, Eps_eri et K3_01 sont tous eps Eri), reconnus par la position J2000 de l'étoile, tirée des coordonnées de la cible dans les en-têtes et ramenée à 2000 par leur mouvement propre. Les deux étoiles d'une binaire restent séparées : par leurs noms (GJ 15 A, GJ 15 B), sinon par SIMBAD, interrogé sur les noms des nouvelles étoiles et des positions qui portent plusieurs noms.\n\nUne étoile qui a déjà un dossier dans la racine le garde ; une nouvelle est nommée comme le dit Nom des nouvelles étoiles, et son dossier peut recevoir un autre nom dans le plan, avant que rien ne bouge. Le mode EGGS de HARPS, le mode HE de SOPHIE et les réglages d'ESPRESSO autres que HR 1x1 ont leurs propres dossiers (<objet>_EGGS). Au passage, les spectres compressés (gzip) sont décompressés, le blaze de SOPHIE est séparé de ses spectres, et les spectres HARPS et HARPS-N dont l'ESO ne fournit pas le blaze en reçoivent un, tiré d'eux.\n\nRien ne bouge avant que vous ayez vu le plan et cliqué Déplacer les fichiers. Rien n'est écrasé : une copie d'un fichier déjà en place est retirée de Downloads, un fichier différent du même nom y reste. Chaque déplacement est noté dans Downloads/.dispatched.jsonl.\n\nLes modèles de chaque dossier d'instrument (spectres PHOENIX, TAPAS, templates validés...) sont ceux de la racine : son models/ est un lien vers RACINE/models/, pour que LBL les télécharge une seule fois pour tous les instruments (un models/ propre au dossier est d'abord fusionné avec celui de la racine, sauf si un de ses fichiers diffère de celui de la racine).",
    "dispatch_headers": "lecture des en-têtes : %s sur %s",
    "dispatch_looking": "examen des dossiers déjà dans la racine",
    "dispatch_simbad": "noms des étoiles demandés à SIMBAD : %s sur %s",
    "dispatch_moving": "déplacement : %s sur %s",
    "dispatch_blazes": "les fichiers sont en place ; fabrication des blazes que l'ESO ne fournit pas, un par nuit : %s sur %s",
    "dispatch_aside": "mise de côté de ce que LBL ne lit pas : %s sur %s",
    "dispatch_tidying": "retrait des dossiers de Downloads restés vides",
    "dispatch_models": "liens des modèles des dossiers d'instrument vers ceux de la racine",
    "dispatch_plan": "%s fichiers dans Downloads : %s spectres dans %s dossiers, %s calibrations, %s mis de côté, %s laissés",
    "dispatch_go": "Déplacer les fichiers",
    "dispatch_cancel": "Annuler",
    "dispatch_new": "nouveau",
    "dispatch_existing": "déjà là",
    "col_dispatch_folder": "Vers",
    "col_dispatch_files": "Spectres",
    "col_dispatch_names": "Noms dans les en-têtes",
    "dispatch_calib": "%s calibrations dans %s/calib/",
    "dispatch_unused": "%s mis de côté dans Downloads/unused/ : %s (p. ex. %s)",
    "dispatch_left": "%s laissés dans Downloads : %s (p. ex. %s)",
    "dispatch_either_star": "Les fichiers nommés %s pourraient être soit %s, où ils vont (le plus proche), soit %s : deux étoiles que SIMBAD distingue",
    "dispatch_simbad_unreachable": "SIMBAD n'a pas répondu : les étoiles sont distinguées par leurs positions et leurs noms seulement.",
    "dispatch_simbad_cut": "SIMBAD a été interrogé sur %s noms au plus : les autres sont reconnus par leur position seulement.",
    "dispatch_nothing": "Rien n'attend dans Downloads.",
    "dispatch_done": "%s fichiers déplacés dans leurs dossiers, %s copies de fichiers déjà en place retirées de Downloads, %s mis de côté dans Downloads/unused/.",
    "dispatch_made": "Blazes tirés des spectres (l'ESO ne les fournit pas) : %s",
    "dispatch_kept": "Laissés dans Downloads, un fichier différent du même nom étant à leur place : %s",
    "dispatch_failed": "Non déplacés : %s",
    "dispatch_error": "La répartition s'est arrêtée : %s",
    "dispatch_share_link": "models/ de %s : chacun devient un lien vers models/ de la racine, une seule copie des modèles de LBL pour tous les instruments",
    "dispatch_share_merge": "models/ de %s : leurs fichiers vont dans models/ de la racine (les copies de fichiers déjà là sont retirées), puis chacun devient un lien vers lui",
    "dispatch_shared": "models/ de %s : liens vers models/ de la racine",
    "dispatch_share_kept": "%s/models reste un dossier à part : %s diffèrent de ceux de la racine",
    "dispatch_close": "Fermer",
    "tree_left": "%s laissés, qui ne peuvent pas être répartis : %s (p. ex. %s)",
    "pick": "Ajouter des fichiers et répartir…",
    "help_pick": "Des spectres de n'importe où sur cette machine, choisis dans la fenêtre du système (plusieurs à la fois : Maj ou Cmd, ou Cmd-A pour tous les fichiers d'un dossier). Ils sont copiés dans un dossier à eux de Downloads (déplacés, si la case est cochée ; un fichier déjà là, identique octet pour octet, n'est pas recopié), puis leurs en-têtes sont lus et le plan montre où va chacun, comme pour Répartir : rien n'entre dans les dossiers d'instrument avant que vous ayez vu le plan et cliqué Déplacer les fichiers.",
    "pick_prompt": "Spectres à ajouter à la racine",
    "pick_waiting": "choisissez les fichiers dans la fenêtre du système…",
    "pick_failed": "la fenêtre du système n'a pas pu s'ouvrir : %s",
    "pick_summary": "%s fichiers choisis (%s), de %s",
    "pick_move": "les déplacer plutôt que les copier (ils quittent leur dossier)",
    "pick_go": "Ajouter et répartir",
    "pick_added": "%s fichiers ajoutés dans %s (%s déjà là)",
    "dispatch_adding": "copie dans Downloads : %s sur %s",
    "dispatch_adding_move": "déplacement dans Downloads : %s sur %s",
    "tab_folders": "Dossiers",
    "tree_title": "Contenu de la racine",
    "root_shown": "racine : %s",
    "explore_planets": "planètes publiées : %s",
    "setup_folder": "%s, d'après le dossier d'instrument %s (le menu du haut)",
    "tree_old": "la racine elle-même, ancienne organisation",
    "tree_old_note": "un seul dossier de données pour tous les instruments ; ses tables et produits restent lisibles (explorateur de données, ménage)",
    "tree_legacy_empty": "%s dossiers vides, aucun spectre",
    "tree_old_folders": "ses dossiers LBL : %s",
    "template_lines": "Raies d'intérêt",
    "template_lines_title": "Les raies dont LBL mesure l'activité (lbl/resources/activity_lines.yaml), à leurs longueurs d'onde dans le vide, ombrées sur la fenêtre où LBL les mesure",
    "template_rest": "Dans le référentiel de l'étoile",
    "template_rest_title": "Le template ramené au référentiel de son étoile (ses longueurs d'onde divisées par le facteur Doppler de v_sys), où les raies d'une espèce sont à leurs longueurs d'onde de laboratoire ; décoché, le template tel que LBL l'a fait et les raies déplacées vers lui",
    "template_vsys": "v_sys (km/s)",
    "template_vsys_title": "La vitesse systémique de l'étoile : le SYSTVELO du masque que LBL a fait à partir du template (mesuré contre un modèle PHOENIX), sinon celle de SIMBAD ; saisissez-en une autre pour l'essayer, videz le champ pour revenir",
    "template_shifted": "ramené au référentiel de l'étoile, v_sys %s km/s (%s)",
    "template_vsys_typed": "saisie",
    "template_vsys_unknown": "aucune vitesse systémique connue (pas de masque, rien dans SIMBAD) : saisissez-en une pour ramener le template au référentiel de son étoile",
    "template_lines_moved": "raies d'intérêt déplacées vers le référentiel du template de v_sys %s km/s",
    "folder_empty": "%s · aucun spectre pour l'instant",
    "tree_summary": "Dans la racine : %s fichiers à répartir dans Downloads · %s dossiers d'instrument, %s cibles, %s spectres",
    "tree_downloads": "%s fichiers à répartir (%s), par nom :",
    "tree_downloads_none": "rien à répartir",
    "tree_unused": "%s fichiers mis de côté, que LBL ne lit pas",
    "tree_home": "%s cibles, %s spectres",
    "tree_home_empty": "aucun spectre pour l'instant",
    "tree_open": "ouvrir",
    "naming": "Nom des nouvelles étoiles",
    "naming_apero": "base d'APERO, sinon SIMBAD",
    "naming_simbad": "nom principal de SIMBAD",
    "naming_headers": "nom des en-têtes (le plus fréquent)",
    "naming_hint": "une étoile déjà dans la racine garde le nom de son dossier : renommez-le dans l'arborescence ci-dessous (✎)",
    "help_naming": "Le nom du dossier que reçoit une étoile nouvelle dans la racine quand elle est répartie :\n\n- Base d'APERO, sinon SIMBAD : le nom que la base astrométrique d'APERO donne à l'étoile (GL699, EPSILONERI, TOI206, AN_SEX...), trouvée par un des noms des en-têtes ou par la position de l'étoile, si bien qu'un fichier HARPS nommé Gl699 ou un surnom de programme (YO17) reçoit lui aussi le nom d'APERO ; les fichiers d'APERO le portent déjà (DRSOBJN). La base est copiée une fois depuis le serveur d'APERO (quelques Mo, gardés dans ~/.cache/lbl/gui). Une étoile qu'APERO ne connaît pas est nommée comme SIMBAD la nomme.\n- Nom principal de SIMBAD : l'identifiant principal de l'étoile dans SIMBAD, sans ses préfixes (* eps Eri donne eps_Eri, NAME Barnard's star donne Barnard_s_star), trouvé par un nom des en-têtes que SIMBAD place là, sinon en demandant à SIMBAD l'étoile la plus brillante là où pointent les fichiers.\n- Nom des en-têtes : le nom que les en-têtes des fichiers donnent le plus souvent à la cible (l'OBJECT de l'observateur, avant le DRSOBJN d'APERO), mis en nom de dossier.\n\nQuand une façon ne donne pas de nom, la suivante le donne, et les en-têtes en dernier. Une étoile qui a déjà un dossier dans la racine garde ce nom, dans chaque dossier d'instrument (une étoile, un nom) : renommez-la dans l'arborescence. Dans le plan, le nom de chaque nouveau dossier peut être tapé à la main avant que rien ne bouge.",
    "col_dispatch_how": "Nom tiré de",
    "how_root": "la racine",
    "how_existing": "son dossier",
    "how_apero": "APERO",
    "how_simbad": "SIMBAD",
    "how_headers": "en-têtes",
    "how_hand": "à la main",
    "dispatch_apero": "lecture de la base d'étoiles d'APERO",
    "dispatch_apero_unreachable": "La base d'étoiles d'APERO n'a pas pu être copiée (son serveur n'a pas répondu) : les nouvelles étoiles sont nommées comme SIMBAD les nomme.",
    "name_edit_title": "Le nom du dossier : lettres, chiffres, + - et _ (Entrée pour le garder, Échap pour annuler)",
    "name_empty": "il faut un nom",
    "name_space_ends": "pas d'espace au début ni à la fin d'un nom",
    "name_chars": "pas dans un nom de dossier : %s (lettres, chiffres, + - et _ seulement ; %s ?)",
    "name_ends": "une lettre ou un chiffre au début et à la fin (%s ?)",
    "name_long": "%s caractères au plus",
    "name_suffix": "%s termine les noms des dossiers que fait LBL : un autre nom",
    "name_same": "le même nom",
    "name_taken": "%s existe déjà : un autre nom",
    "rename_missing": "%s n'est pas là (plus là ?)",
    "rename_busy": "un calcul travaille dans la racine (processus %s) : renommez une fois qu'il est fini",
    "rename_nothing_new": "les fichiers de cette étoile vont dans un dossier déjà là : renommez-le dans l'arborescence",
    "rename_no_plan": "le plan n'est plus affiché : répartissez de nouveau",
    "tree_rename": "Renommer ce dossier",
    "rename_ok": "Renommer",
    "rename_everywhere": "aussi ses dossiers du même nom dans %s",
    "rename_will": "Renomme %s en %s.",
    "rename_products": "%s : %s fichiers et dossiers que LBL a faits sous le nom %s le gardent (p. ex. %s) : le prochain calcul les refait sous le nouveau nom, et l'onglet Ménage peut effacer les anciens.",
    "rename_done": "%s → %s : %s dossiers renommés",
    "tree_current": "en cours",
    "tree_empty": "%s dossiers vides, pas des cibles : %s",
    "tree_calib": "%s fichiers",
    "tree_models_link": "-> ../models, celui de la racine",
    "tree_models_own": "un dossier à part (la prochaine répartition le fusionne avec celui de la racine)",
    "tree_models_none": "aucun pour l'instant (la prochaine répartition le lie à celui de la racine)",
    "tree_models": "partagé par tous les dossiers d'instrument : %s fichiers, %s",
    "tree_others": "Aussi dans la racine : %s",
    "targets_empty_hidden": "%s dossiers vides non montrés (aucun spectre) : %s",
    "dispatch_default_blaze": "%s : pas encore de blaze par défaut, le premier blaze de l'ESO y est mis (models/default-blaze/), que LBL lit pour les spectres dont le blaze de la nuit manque",
    "dispatch_defaults": "Blaze par défaut mis en place : %s",
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
    "tab_explore": "Explorateur",
    "explore_title": "Explorateur de données",
    "explore_rdb": "Étoile et saveur (lblrdb)",
    "explore_x": "x",
    "explore_y": "y",
    "explore_c": "Couleur",
    "explore_nightly": "par nuit (lbl2)",
    "help_explore_nightly": "LBL écrit deux tables à chaque exécution : lbl_, une ligne par spectre, et lbl2_, les spectres de chaque nuit combinés en une ligne. Les menus donnent la première ; cochée, la case montre les graphiques et le FIP de la seconde, de la même exécution, pour chaque table qui en a une. Grisée quand aucune n'en a.",
    "help_explore": "Les étoiles des tables rdb que LBL a écrites dans DATA_DIR/lblrdb, la plus récente en premier, et pour chacune ses saveurs : ce que le nom d'une table dit d'autre, la variante PCA2D et le template (pca2d / GJ_1028_pca2d, VETTED_GL880...). Une ligne par spectre (la table des nuits, lbl2_, avec la case à côté des menus). N'importe quelle colonne numérique contre n'importe quelle autre. Les barres d'erreur sont celles de la colonne d'erreur de la colonne quand elle en a une (vrad : svrad, fwhm : sig_fwhm, DTEMP3500 : sDTEMP3500) ; la couleur est une troisième colonne, ou la probabilité de chaque point d'être aberrant, tirée du FIP robuste aux points aberrants de y (bleu fiable, gris indécis, rouge aberrant).\n\nGlissez pour zoomer, double-cliquez pour tout revoir ; le fichier d'un point s'affiche sous la souris. Une nouvelle fenêtre s'ouvre sur la table la plus récente, en x le MJD, en y la vitesse, en couleur la BERV.\n\nÀ droite, sur le même axe y, l'histogramme de y (de chaque table en fraction de ses points quand il y en a plusieurs). Sous le graphique, pour chaque table, le rms de y (l'écart-type) et son rms robuste (l'écart absolu médian fois 1,4826, le rms d'une gaussienne, que quelques points aberrants ne déplacent pas), comme le rapport les donne, et sa barre d'erreur médiane.\n\n+ superpose une autre saveur de la même étoile (jusqu'à 8 ; une autre étoile quand toutes sont montrées, ou par son menu) : les tables prennent alors une couleur chacune, le y de chacune est pris moins sa médiane pour qu'elles se superposent, et les menus offrent les colonnes qu'elles ont toutes. Le x à côté d'une table la retire.",
    "explore_fip": "Probabilité de fausse inclusion",
    "help_explore_fip": "La probabilité de fausse inclusion (FIP, Hara et al. 2022) de y, et de x quand x n'est pas un temps, contre le temps de la table (rjd), sur des périodes de 1,1 jour à deux fois sa durée : la probabilité qu'une sinusoïde à cette période ne soit PAS dans les données, en -log10 (2 vaut 1 %, en pointillé ; plus c'est haut, plus c'est sûr).\n\nGaussien (tirets) : chaque point pris à sa barre d'erreur. Robuste aux points aberrants (trait plein) : chaque point compte pour sa probabilité d'être normal, pour qu'une mauvaise nuit n'invente ni n'enterre un signal ; là où les deux se séparent, la réponse dépend du traitement des points aberrants. Chaque modèle ajuste son propre jitter. Ce sont ceux du rapport (lbl/report/fip.py). Une colonne sans colonne d'erreur est prise à sa dispersion robuste. Avec plusieurs tables, le FIP du y de chacune, dans sa couleur (le gaussien en tirets).\n\nLes lignes en tirets-points sont les périodes des planètes publiées de l'étoile, nommées : celles des catalogues que lit l'option PLANETS de LBL (lbl/resources/planets : exoplanet.eu et le NASA Exoplanet Archive, après un personal_planet_catalog.yaml à vous), y compris celles sans orbite complète.",
    "star_heading": "L'étoile",
    "help_explore_star": "L'étoile de chaque table montrée, une fiche par étoile (son nom la replie). De SIMBAD (au CDS, sinon à ses miroirs du CfA et de l'IDIA) : ses noms, son type, son type spectral, sa position, sa parallaxe, son mouvement propre et sa vitesse systémique, V G J H Ks, les médianes des Teff, [Fe/H], log g et v sin i qu'il recense. De Gaia DR3 (l'archive de l'ESA, sinon la copie du GAVO) : BP, RP, parallaxe, mouvement propre, RUWE et, de l'ESA seulement, les paramètres GSP-Phot et FLAME. D'AllWISE (GAVO, sinon IRSA) : W1 et W2 de la source la plus proche de l'étoile ramenée à 2010,5.\n\nDérivé comme le fait le star-o-meter. Une naine M (M_G de 7,5 à 16,5, ou de type M) : [Fe/H] de W1-W2, BP-RP et M_G (Duque-Arribas et al. 2023, éq. 4 ; pour -0,10 <= W1-W2 <= 0,24 et 1,85 <= G-J <= 3,10 : le W1 saturé d'une étoile brillante n'en donne pas), Teff de BP-RP avec J-H ou [Fe/H] (Mann et al. 2015 ; 2700 à 4100 K), R et M de M_Ks (Mann et al. 2015 ; 4,6 < M_Ks < 9,8 ; R avec [Fe/H] multiplié par 1 + 0,04458 [Fe/H]), M de M_K (Delfosse et al. 2000 ; 4,5 à 9,5), log g et L à partir d'eux. Une naine AFGK : GSP-Phot et FLAME de Gaia, sinon la séquence des naines de Pecaut & Mamajek (2013) à son BP-RP et son M_G, ou à son type spectral. Rien pour une géante, une naine blanche ou une étoile O ou B. Une relation hors de son domaine ne donne rien, et dit pourquoi.\n\nPublié : les médianes de SIMBAD, et les paramètres de l'hôte du NASA Exoplanet Archive, chacun avec son article. Planètes : la table composite de l'archive (pscomppars), trouvées par les numéros Gaia DR3, TIC, HD et HIP de l'étoile et ses noms : période, K, masse (« sin i » quand seule celle-là est connue), rayon, demi-grand axe, excentricité, température d'équilibre, comment et quand trouvées, les articles de la découverte et de l'orbite. Quand l'archive ne répond pas, les planètes du catalogue de LBL (période, K).\n\nLes telluriques : les jours de l'année où la vitesse de l'étoile vue du centre de la Terre, RV - BERV (composées de façon relativiste), est sous 5 km/s : ses raies tombent alors sur les raies telluriques. Le BERV omet la rotation de la Terre (moins de 0,5 km/s).\n\nDemandé une fois, gardé dans DATA_DIR/gui/star_info.json pendant 30 jours ; Redemander les interroge de nouveau.",
    "star_asking": "SIMBAD, Gaia, AllWISE et le NASA Exoplanet Archive sont interrogés...",
    "star_refresh": "Interroger de nouveau SIMBAD, Gaia, AllWISE et le NASA Exoplanet Archive",
    "star_again": "Redemander",
    "star_unknown": "SIMBAD ne connaît aucun de ces noms : %s",
    "star_failed": "N'a pas pu être interrogé : %s",
    "star_n_planets": "%s planètes publiées",
    "star_one_planet": "1 planète publiée",
    "star_unreached": "sans réponse",
    "star_no_planet": "aucune planète publiée",
    "star_g_position": "Position et mouvement",
    "star_g_photometry": "Photométrie",
    "star_g_derived": "Dérivé : %s",
    "star_g_published": "Publié",
    "star_mode_mdwarf": "naine M",
    "star_mode_afgk_gaia": "naine AFGK, Gaia DR3",
    "star_mode_afgk_seq": "naine AFGK, séquence des naines",
    "star_mode_wd": "naine blanche, rien",
    "star_mode_other": "pas une naine, rien",
    "star_mode_early": "étoile O ou B, rien",
    "star_mode_none": "rien",
    "star_published_none": "rien de recensé",
    "star_median": "médiane de %s",
    "star_note_out": "%s = %s est hors de %s à %s (%s) : non dérivé.",
    "star_note_duque_missing": "[Fe/H] : W1, W2, BP, RP ou la parallaxe manque.",
    "star_note_no_bprp": "Teff : pas de BP-RP de Gaia.",
    "star_note_no_mk": "R et M : pas de Ks ou pas de parallaxe.",
    "star_note_sequence": "Sur la séquence des naines près de %s.",
    "star_tellurics": "Raies sur les telluriques (|RV - BERV| < %s km/s) :",
    "star_tellurics_never": "jamais",
    "star_tellurics_range": "RV %s km/s, BERV de %s à %s km/s en %s",
    "star_tellurics_no_rv": "Raies sur les telluriques : aucune vitesse systémique connue.",
    "star_window": "du %s au %s",
    "star_planets": "Planètes publiées (NASA Exoplanet Archive)",
    "star_planets_lbl": "Planètes (catalogue de LBL : le NASA Exoplanet Archive n'a pas répondu)",
    "star_planets_none": "Aucune dans le NASA Exoplanet Archive.",
    "star_planets_none_lbl": "Aucune dans le catalogue de LBL (le NASA Exoplanet Archive n'a pas répondu).",
    "star_pl_name": "planète",
    "star_pl_period": "P [j]",
    "star_pl_k": "K [m/s]",
    "star_pl_mass": "M [M⊕]",
    "star_pl_radius": "R [R⊕]",
    "star_pl_a": "a [ua]",
    "star_pl_e": "e",
    "star_pl_teq": "Téq [K]",
    "star_pl_method": "trouvée",
    "star_pl_refs": "articles",
    "star_pl_controversial": "signalée comme controversée par l'archive",
    "star_sources": "SIMBAD : %s · Gaia DR3 : %s · AllWISE : %s · planètes : %s · demandé le %s",
    "star_nasa_down": "catalogue de LBL",
    "template_explorer": "Explorateur de template",
    "help_template_explorer": "Le template contre lequel la table a été mesurée (DATA_DIR/templates, ou models/template_* pour un template vetted) et, sous lui, quatre fois moins haut et sur les mêmes longueurs d'onde, la transmission TAPAS de chaque molécule de l'atmosphère à une masse d'air de 1, de 1 (transparent) à 0 (opaque).\n\nGlissez sur l'un ou l'autre pour zoomer en longueur d'onde (le flux suit), double-cliquez pour tout le template ; la molette zoome aussi. En zoom serré, chaque point du template est tracé ; de loin, le plus bas et le plus haut de chaque pixel de l'écran (la transmission la plus basse pour TAPAS). Un clic sur une molécule de la légende la cache.\n\nLe fichier TAPAS est celui de DATA_DIR/models/tellurics qui a une colonne par molécule (wavelength, H2O, O2, CH4, CO2, O3, N2O, NO2), celui qui porte le nom du site de l'instrument s'il y en a plusieurs (LaSilla, MaunaKea...), sinon le tapas_lbl.fits de LBL, qui a l'eau et les autres molécules ensemble. TAPAS est dans le référentiel de l'observateur et le template dans le référentiel barycentrique : sur le template, chaque raie tellurique bouge avec la BERV de chaque spectre.\n\nAvec plusieurs tables, le template de chacune dans sa couleur (un seul pour les tables qui le partagent), chacun divisé par sa médiane sur les longueurs d'onde montrées ; TAPAS est celui de l'instrument de la première table.\n\nLe template est montré dans le référentiel de son étoile, déplacé de la vitesse systémique du masque que LBL en a tiré (SYSTVELO, mesurée contre un modèle PHOENIX ; sinon celle de SIMBAD, ou celle que vous saisissez) : les raies d'intérêt (activity_lines.yaml, dans le vide) sont ombrées là où LBL les mesure. Décoché, le template est tel que LBL l'a fait et les raies sont déplacées vers lui. Une boîte coupe x et y ; une boîte plate, un défilement en x ou un déplacement gardent y ajusté à ce qui est vu : deux fois l'écart entre ses 3e et 97e centiles autour de leur milieu, sans dépasser ses extrêmes, pour que quelques points aberrants ne fixent pas l'échelle ; un double clic montre tout.",
    "explore_no_plotly": "Les graphiques de cet onglet sont tracés par Plotly, que la page va chercher sur cdn.jsdelivr.net et n'a pas pu obtenir : cette machine est-elle hors ligne ? Les autres onglets s'en passent.",
    "explore_no_rdb": "Pas encore de table rdb dans %s : elles sortent de l'étape compile de LBL.",
    "explore_unreadable": "N'a pas pu être lu : %s.",
    "explore_add": "+ superposer",
    "explore_hist_count": "N",
    "explore_hist_fraction": "fraction",
    "explore_col_table": "table",
    "explore_col_n": "points",
    "explore_col_rms": "rms",
    "explore_col_robust": "rms robuste",
    "explore_col_sigma": "erreur médiane",
    "explore_remove": "retirer cette table des graphiques",
    "explore_minus_median": "%s moins sa médiane",
    "explore_colour_tables": "une couleur par table, le y de chacune moins sa médiane",
    "explore_colour_none": "aucune",
    "explore_outlier": "P(aberrant), FIP robuste de y",
    "explore_outlier_short": "P(aberrant)",
    "explore_rows": "%s lignes",
    "explore_bars": "barres d'erreur de y : %s",
    "explore_no_bars": "%s n'a pas de colonne d'erreur : pas de barres d'erreur",
    "explore_outlier_wait": "les probabilités d'être aberrant viennent avec le FIP",
    "explore_fip_no_time": "Cette table n'a pas de colonne rjd : pas de FIP.",
    "explore_fip_wait": "calcul du FIP de %s...",
    "explore_fip_none": "moins de 10 points avec une erreur : pas de FIP",
    "explore_gauss": "gaussien",
    "explore_oa": "robuste",
    "explore_fip_best": "%s : FIP robuste le plus bas à %s j (%s) ; %s points sur %s plus probablement aberrants que non",
    "explore_fip_noerr": "pas de colonne d'erreur : prise à sa dispersion robuste",
    "explore_year": "1 an",
    "explore_period_axis": "Période (jours)",
    "explore_fip_axis": "-log10 FIP (1 % en pointillé)",
    "template_none": "Aucun template trouvé pour cette table, dans DATA_DIR/templates ou models/template_*.",
    "template_flux": "Flux du template",
    "template_flux_median": "Flux du template / sa médiane",
    "template_several": "templates : %s",
    "template_missing": "aucun template trouvé pour %s",
    "template_transmission": "Transmission",
    "template_transmission_short": "transmission",
    "template_others": "autres",
    "template_wavelength": "Longueur d'onde (nm)",
    "template_of": "template : %s",
    "template_tapas": "TAPAS : %s, masse d'air 1, référentiel de l'observateur",
    "template_tapas_lbl": "TAPAS : %s (l'eau, et les autres ensemble), masse d'air 1, référentiel de l'observateur",
    "template_berv": "les raies telluriques bougent sur le template avec la BERV, ici de %s à %s km/s",
    "template_no_tapas": "pas de fichier TAPAS dans DATA_DIR/models/tellurics",
    "tab_refs": "Références",
    "tab_docs": "Documentation",
    "docs_search": "Chercher dans la documentation",
    "docs_toc": "Sur cette page",
    "docs_english": "Ces pages sont en anglais, comme la documentation de LBL ; les textes d'aide (i) des champs y sont repris dans leur version anglaise.",
    "docs_prev": "← %s",
    "docs_next": "%s →",
    "docs_none": "Rien trouvé pour %s.",
    "docs_file": "Cette page : %s",
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
    "sw_PLANETS": "planètes connues (leurs orbites)",
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
    "col_basis": "base PCA2D",
    "freeze_basis": "Geler la base",
    "freeze_title": "Garder la base de l'observateur de ce run (%s) sous un nom, dans models/pca2d_<INSTRUMENT>_<SOURCE>/ : les prochains runs PCA2D de cet instrument et de cette source peuvent l'utiliser gelée (input_name), et aucun ménage ne l'efface",
    "freeze_placeholder": "un nom : nirps_2026a",
    "freeze_go": "Geler",
    "freeze_will": "Gardée sous %s ; les réglages PCA2D de cet instrument et de cette source la proposent sous le nom %s.",
    "freeze_done": "Base gelée sous le nom %s : %s",
    "basis_frozen_as": "gelée sous le nom %s",
    "basis_used": "a lu la base gelée %s",
    "basis_replaced": "remplacée par un fit plus récent de %s",
    "basis_missing": "pas de base sur le disque",
    "basis_running": "une fois le run fini",
    "freeze_taken": "%s est le nom d'une base déjà là (%s) : un autre nom",
    "freeze_target": "%s est une cible de ce dossier : son prochain fit écrirait par-dessus la base gelée ; un autre nom",
    "freeze_already": "cette base est déjà gelée, sous le nom %s",
    "freeze_no_source": "%s ne dit pas son instrument ni sa source de données (INSTRUME, DATASRC) : elle ne peut pas être rangée",
    "freeze_state_used": "ce run a lu une base gelée : il n'en a pas fitté",
    "freeze_state_replaced": "un fit plus récent a écrit par-dessus la base de ce run",
    "freeze_state_missing": "la base de ce run n'est plus sur le disque",
    "freeze_state_running": "le run n'est pas fini",
    "freeze_state_none": "ce run n'a fitté aucune base PCA2D",
    "col_object": "objet",
    "col_report": "rapport",
    "col_date": "date",
    "nothing": "rien pour l'instant",
    "measure": "Mesurer à nouveau",
    "busy": "Un run tourne dans ce dossier : rien ne peut être supprimé avant qu'il se termine. %s",
    "busy_unknown": "Ce système ne permet pas de savoir si un run tourne dans ce dossier (ni psutil ni lsof ne s'y trouvent) : rien ne peut donc être supprimé. Installez psutil (pip install psutil).",
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
    "what_residuals": "résidus PCA2D de chaque pose (refaits à partir du fit, avec save_residuals)",
    "what_cleaned": "spectres nettoyés des telluriques",
    "what_lblrv": "vitesses de LBL par pose (une dizaine de minutes à refaire)",
    "what_template": "un template LBL",
    "what_mask": "un masque LBL",
    "what_rdb": "les tables de vitesses de LBL",
    "what_reports": "les rapports",
    "what_components": "bases de l'observateur et composantes des étoiles",
    "what_models": "les modèles de LBL : modèles stellaires, TAPAS, templates VETTED, bases PCA2D gardées",
    "what_downloads": "téléchargé, en attente de répartition",
    "what_unused": "produits que LBL ne lit pas, mis de côté par la répartition (CCF, S1D...)",
    "explore_root_home": "racine, ancienne organisation",
    "lbl_says": "La description de LBL (en anglais) :",
    "help_data_dir": "La racine de vos données LBL. Mettez ce que vous téléchargez dans son dossier Downloads/, et Répartir le classe : chaque spectre dans RACINE/{INSTRUMENT}_{SOURCE}/science/{OBJET}/ (RACINE/{INSTRUMENT}/ pour un instrument que LBL lit d'une seule source : ESPRESSO, SOPHIE...), avec les noms que LBL donne aux instruments et aux sources de données. Chacun de ces dossiers est un dossier de données LBL (DATA_DIR) : le wrap le reçoit, et LBL y écrit ses templates, masques, vitesses par pose (lblrv), tables (lblrdb) et rapports.\n\nUn dossier qui contient lui-même science/<objet>/ (l'ancienne organisation) fonctionne toujours : choisissez « (ce dossier) » dans le menu dessous.\n\n« Lire les cibles » lit une fois l'entête de chaque spectre (SNR, BERV, date, Teff) et garde le résultat : une relecture plus tard est rapide.",
    "help_instrument": "L'instrument LBL, et son mode, dont viennent les spectres. Il fixe la façon dont LBL lit les fichiers, le domaine de longueurs d'onde et la résolution. Deviné à partir des entêtes à la lecture des cibles ; NIRPS_HE et NIRPS_HA sont les deux modes de NIRPS.",
    "help_source": "Le pipeline qui a réduit les spectres (DATA_SOURCE : APERO, ESO, CADC...). Il fixe le format de fichier qu'attend LBL. Une base PCA2D faite à partir d'une autre source ne peut pas servir ici.",
    "help_python": "Le Python qui lance les wraps : celui de l'environnement où LBL est installé (les étapes d'installation l'expliquent). Par défaut, le Python de cette fenêtre.",
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
    "help_sw_RUN_LBL_TEMPLATE": "Construire le template de l'étoile : tous ses spectres ramenés dans le référentiel de l'étoile et combinés en un spectre à haut SNR (templates/). Aucun spectre n'est rééchantillonné : chaque pixel de chaque spectre garde sa propre longueur d'onde, les spectres ensemble échantillonnent l'étoile bien plus finement que le détecteur, et le template et ses dérivées sont un polynôme local ajusté à travers eux, sur un élément de résolution (lbl/doc/TEMPLATE.md). Chaque raie est mesurée contre lui. Seulement pour une cible mesurée contre son propre template ; celui d'une autre étoile est lu tel quel.",
    "help_sw_RUN_LBL_MASK": "Construire la liste de raies (masks/) à partir du template : la position et le signe de chaque raie, les domaines sur lesquels les vitesses sont mesurées, et la vitesse systémique de l'étoile.",
    "help_sw_RUN_LBL_COMPUTE": "Mesurer la vitesse de chaque raie dans chaque pose contre le template (lblrv/, un fichier par pose). La plus longue des étapes de LBL, typiquement une dizaine de minutes.",
    "help_sw_RUN_LBL_COMPILE": "Combiner les vitesses par raie de chaque pose en une vitesse et son erreur, avec les indicateurs (d2v, d3v, DTEMP...), dans les tables rdb de lblrdb/ : lbl_ par pose, lbl2_ par nuit.",
    "help_sw_SKIP_LBL_TEMPLATE": "Garder un template déjà sur disque plutôt que de le refaire. À décocher quand les spectres ont changé (plus nombreux, une autre correction) : les vitesses doivent être mesurées contre un template fait des mêmes spectres. Avec PCA2D, une nouvelle correction met d'elle-même les anciens produits de côté.",
    "help_sw_SKIP_LBL_MASK": "Garder un masque déjà sur disque plutôt que de le refaire. À décocher quand le template a été refait.",
    "help_sw_SKIP_LBL_COMPUTE": "Ne pas remesurer une pose dont les vitesses par raie sont sur disque : seules les nouvelles poses sont mesurées. À décocher quand le template ou le masque a changé, sinon d'anciennes et de nouvelles vitesses se mélangent dans une même table.",
    "help_sw_SKIP_LBL_COMPILE": "Garder les tables rdb sur disque. Désactivé par défaut : compiler prend quelques secondes et doit inclure chaque nouvelle pose.",
    "help_sw_REPORT": "Écrire le rapport PDF dans reports/ : les vitesses, leurs périodogrammes, les indicateurs et, avec PCA2D, ses propres pages d'abord (les river plots, les composantes, les vitesses avant et après).",
    "help_sw_PLANETS": "Utiliser les planètes connues de l'étoile : les orbites publiées de ses planètes confirmées (le catalogue de LBL, lbl/resources/planets/known_planets.yaml), ou vos propres solutions (DATA_DIR/personal_planet_catalog.yaml), qui les remplacent pour cette étoile. Le template est construit sur le BERV plus leur vitesse képlérienne, et le rapport ajuste leur K à leurs périodes. Les vitesses rapportées restent corrigées du seul BERV.",
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
    "err_basis_grid": "la base %s n'est pas sur la grille de ce run, IL FAUT LA RECONSTRUIRE (relancer le fit conjoint qui l'a faite, avec ce LBL) ; LBL la refuse : %s",
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
    "help_steps": "Les étapes du run d'un bout à l'autre, telles que le wrap les prévoit : pour chaque cible, LBL sur les spectres livrés (quand PCA2D les mesure d'abord), le cube, le fit, les figures et la correction de PCA2D, puis LBL sur les spectres corrigés et le rapport. Chacune se coche quand le journal atteint la suivante ; celle en cours est encadrée, une étape en échec est en rouge. Chaque étape qui tourne a sa pastille : un template ou un masque dont l'étape est décochée mais qui n'est pas sur le disque est fait quand même avant les vitesses, et sa pastille est en pointillés (son info-bulle dit pourquoi) ; une étape que le journal nomme et que le plan n'avait pas reçoit une pastille en pointillés là où elle vient.",
    "step_telluclean": "nettoyage tellurique",
    "step_telluclean_1": "1er nettoyage tellurique, à l'aveugle",
    "step_template_tc": "template des spectres nettoyés",
    "step_telluclean_2": "2e nettoyage tellurique, divisé par ce template",
    "step_template_tc2": "template du 2e nettoyage",
    "step_template": "template",
    "step_mask": "masque",
    "step_template_needed": "template",
    "step_mask_needed": "masque",
    "step_added": "Pas dans le plan : le journal dit que le run l'a fait (un template ou un masque qui manquait, fait avant les vitesses)",
    "step_needed_tip": "Son étape est décochée dans les réglages LBL, mais il n'est pas sur le disque et les vitesses en ont besoin : le run le fait quand même",
    "step_compute": "vitesses",
    "step_compile": "table rdb",
    "step_report": "rapport",
    "step_pca2d_cube": "cube PCA2D",
    "step_pca2d_fit": "fit PCA2D",
    "step_pca2d_figures": "figures PCA2D",
    "step_pca2d_correct": "correction PCA2D",
    "no_data_dir": "Choisissez une racine de données ci-dessus : le dossier qui contient Downloads/ et les dossiers d'instrument.",
    "no_science": "%s n'existe pas. Un dossier de données contient un dossier par cible dans science/ : science/<objet>/, avec les spectres de cette cible.",
    "empty_science": "%s ne contient aucun spectre : chaque cible est un dossier science/<objet>/ avec ses spectres.",
    "read_of": "entêtes lus : %s sur %s",
    "err_mixed_title": "Deux instruments cochés.",
    "err_mixed": "Un run mesure les spectres d'un seul instrument (%s) : décochez les cibles de l'un des deux.",
    "keep_family": "Garder %s, décocher les autres",
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
    "err_folder_source": "%s contient les spectres de %s %s : choisissez cet instrument et cette source de données, ou le dossier d'instrument de celle choisie",
    "err_layout_source": "%s contient des spectres au format de la source de données %s, et celle choisie est %s : choisissez %s",
    "help_archive": "Les spectres publics que LBL lit, d'une étoile ou de plusieurs (séparez leurs noms par des virgules), dans les archives que la fenêtre connaît : les t.fits de SPIRou au CADC, NIRPS, HARPS et ESPRESSO à l'ESO, HARPS-N au TNG, SOPHIE à l'OHP. Chaque nom est résolu par SIMBAD (son nom, ses coordonnées et sa Teff s'affichent sous le champ), puis chaque étoile est demandée à chaque archive en même temps, deux requêtes à la fois par archive. Une archive dont le télescope ne voit jamais l'étoile (moins de 20 degrés au-dessus de son horizon au mieux) n'est pas interrogée, et l'étoile y est dite « hors de portée » ; une archive qui ne répond pas est signalée, et Arrêter la recherche cesse de l'attendre.\n\nLes lignes cochées sont récupérées dans le dossier de la racine de leur instrument et de leur source de données, {INSTRUMENT}_{SOURCE}/science/<étoile>/ (le dossier que l'étoile y a déjà, s'il existe ; les calibrations dans son calib/), quatre fichiers à la fois, les instruments en même temps, sans rien déplacer ni écraser ; un dossier qui contient les spectres d'un autre lecteur en reçoit un à part. Chaque dossier garde sa provenance, et le wrap en tire la source des données et le motif des fichiers. Le journal sous les résultats dit ce que chaque archive a répondu et pourquoi un fichier n'a pas été récupéré.",
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
         "1 gave the better velocities on every star where the two were"
         " compared. The fit's folder is named s<n_star>-o<n_earth>.",
         "Le nombre de spectres qui décrivent l'étoile, son template"
         " compris.\n\n"
         "1 (le nominal) : l'étoile est son template seul, porté au BERV de"
         " chaque pose et retiré avec un coefficient d'exactement 1, sans"
         " ajustement de profondeur. 2 ajoute une composante dans le"
         " référentiel de l'étoile, avec une amplitude libre par pose, pour"
         " regarder la variabilité propre de l'étoile ; elle n'est jamais"
         " divisée des fichiers corrigés. 0 est refusé.\n\n"
         "1 a donné les meilleures vitesses sur toutes les étoiles où les"
         " deux ont été comparés. Le dossier du fit est nommé"
         " s<n_star>-o<n_earth>.")),
    "n_earth": (
        ("observer components", "composantes de l'observateur"),
        ("How many components fixed in the observer's frame are fitted:"
         " telluric residuals, airglow, detector features, everything that"
         " stays put on the detector while the star moves with the BERV."
         " They are divided out of the spectra LBL then measures.\n\n"
         "More components follow more of the atmosphere, but also have more"
         " freedom to take part of the star when the BERV moves little. The"
         " nominal is the instrument's own (LBL's PCA2D_N_EARTH), shown as"
         " the default here.\n\n"
         "With a frozen basis (below), leave it unset to take all of its"
         " components.",
         "Le nombre de composantes fixes dans le référentiel de"
         " l'observateur : résidus telluriques, airglow, défauts du"
         " détecteur, tout ce qui reste en place sur le détecteur pendant"
         " que l'étoile se déplace avec le BERV. Elles sont divisées des"
         " spectres que LBL mesure ensuite.\n\n"
         "Plus de composantes suivent mieux l'atmosphère, mais ont aussi plus"
         " de liberté pour absorber une partie de l'étoile quand le BERV"
         " varie peu. Le nominal est celui de l'instrument (PCA2D_N_EARTH"
         " de LBL), affiché ici comme défaut.\n\n"
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
         " telluric weights).\n\n"
         "Only the measurement of the amplitudes changes: what is divided"
         " out, the blanked samples and the shrinkage do not. The two gave"
         " indistinguishable velocities where they were compared; velocity"
         " is nominal on principle.",
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
         " et poids telluriques).\n\n"
         "Seule la mesure des amplitudes change : ce qui est divisé, les"
         " échantillons masqués et le rétrécissement restent identiques. Les"
         " deux ont donné des vitesses indiscernables là où ils ont été"
         " comparés ; velocity est le nominal par principe.")),
    "mask": (
        ("samples blanked in the corrected files",
         "échantillons masqués dans les fichiers corrigés"),
        ("Which samples the corrected files blank (NaN), where the fit had"
         " nothing to say.\n\n"
         "common (the nominal): a sample is blanked in every exposure as"
         " soon as the fit gave it no weight in one of them (see the share"
         " below), so the whole campaign is measured on ONE set of lines."
         " exposure: each exposure blanks its own, so every epoch has a"
         " different set of lines, which alone adds scatter to the"
         " velocities (1.8 m/s of rms in one comparison). none: keep the"
         " delivered flux there.",
         "Quels échantillons les fichiers corrigés masquent (NaN), là où le"
         " fit n'avait rien à dire.\n\n"
         "common (le nominal) : un échantillon est masqué dans toutes les"
         " poses dès que le fit ne lui a donné aucun poids dans l'une"
         " d'elles (voir la fraction plus bas) ; toute la campagne est donc"
         " mesurée sur UN seul jeu de raies. exposure : chaque pose masque"
         " les siens, et chaque époque a un jeu de raies différent, ce qui"
         " suffit à disperser les vitesses (1.8 m/s de rms dans une"
         " comparaison). none : garder le flux livré à ces endroits.")),
    "mask_share": (
        ("share of exposures that must weight a sample",
         "fraction des poses qui doivent pondérer un échantillon"),
        ("With mask = common, the share of the exposures that must have"
         " given a sample weight for it to be kept.\n\n"
         "1 (the nominal) is unanimity: the strictest, and cheap when every"
         " exposure covers the same range. Below 1 the set of lines is"
         " decided by a majority, still one set for the whole campaign;"
         " useful when the exposures cover different ranges (for a star"
         " observed over a wide range of BERV, unanimity can blank close to"
         " half the pixels once each exposure is shifted by its own BERV).",
         "Avec mask = common, la fraction des poses qui doivent avoir donné"
         " du poids à un échantillon pour qu'il soit gardé.\n\n"
         "1 (le nominal) est l'unanimité : la plus stricte, et peu coûteuse"
         " quand toutes les poses couvrent le même intervalle. Sous 1, le"
         " jeu de raies est décidé à la majorité, et reste unique pour toute"
         " la campagne ; utile quand les poses couvrent des intervalles"
         " différents (pour une étoile observée sur un large intervalle de"
         " BERV, l'unanimité peut masquer près de la moitié des pixels une"
         " fois chaque pose décalée de son propre BERV).")),
    "nsig_cut": (
        ("residual clip (sigmas)", "coupure du résidu (sigmas)"),
        ("Blank every sample whose residual (the data less everything the"
         " fit models, panel 5 of the river plots) is beyond this many"
         " robust sigmas, measured locally (1.4826 MAD over the high pass's"
         " window): cosmic rays, bad pixels, local sky residuals.\n\n"
         "Together with the bias test below, it lowered the rms of a"
         " planet host's velocities from 17.0 to 14.6 m/s without moving the"
         " planet's K. Unset: nothing clipped.",
         "Masquer tout échantillon dont le résidu (les données moins tout ce"
         " que le fit modélise, panneau 5 des river plots) dépasse ce nombre"
         " de sigmas robustes, mesurés localement (1.4826 MAD sur la fenêtre"
         " du passe-haut) : rayons cosmiques, mauvais pixels, résidus de"
         " ciel locaux.\n\n"
         "Avec le test de biais plus bas, il a fait passer le rms des"
         " vitesses d'une étoile à planète de 17.0 à 14.6 m/s, sans déplacer"
         " le K de la planète. Vide : rien n'est coupé.")),
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
         " (methane and water near 1761 to 1805 nm). 10 sigmas typically"
         " blanks about 10% of the domain; lower blanks more.",
         "Une colonne dont le résidu MOYEN sur toutes les poses (sommé sur"
         " deux éléments de résolution) dépasse ce nombre de sigmas est"
         " masquée dans toutes les poses.\n\n"
         "La coupure du résidu compare chaque pose aux autres : elle ne voit"
         " pas un résidu présent dans toutes. La moyenne sur N poses gagne"
         " √N et révèle des raies entières mal corrigées (méthane et eau"
         " vers 1761 à 1805 nm). 10 sigmas masquent typiquement environ"
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
         " and is taken out before this test. Off (unset) by default: it has"
         " been measured on too few stars for a nominal value.",
         "Le test des colonnes ci-dessus, dans le référentiel de l'étoile :"
         " chaque pose portée par son propre décalage, une colonne de"
         " l'étoile où plus de cette fraction des poses a été coupée (et"
         " dont les poses restantes dépassent encore le chi² suivant) est"
         " masquée là où elle tombe, dans toutes les poses. Il attrape ce"
         " qui reste immobile sur l'étoile, comme le laisse un intervalle de"
         " BERV étroit. Mesurée contre le template d'une autre étoile,"
         " la différence statique entre les deux étoiles (leur métallicité)"
         " est attendue et retirée avant ce test. Désactivé (vide) par"
         " défaut : il a été mesuré sur trop peu d'étoiles pour fixer une"
         " valeur nominale.")),
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
    "save_residuals": (
        ("save residuals", "sauver les résidus"),
        ("Also write the residual of every exposure, its star and observer"
         " components removed, into science/<target>_residuals/"
         " (science/<target>_residuals_<name>/ for a named correction): one"
         " file per exposure, in the format of its source (a t.fits for"
         " APERO) and under its name, whose flux is replaced by the log"
         " residual of the fit, ln f - savgol(ln f) - model. Zero where the"
         " model explains the data, -0.01 where the flux is 1 % below it,"
         " NaN where the fit gave the sample no weight; the other extensions"
         " (wavelengths, blaze) as they were. Not written when the fit's"
         " rows are nights or bundles (no exposure has a residual of its"
         " own). Off by default. Asked of a correction already on disk"
         " without them, the correction is made again to write them.",
         "Écrire aussi le résidu de chaque pose, ses composantes stellaires"
         " et observateur retirées, dans science/<cible>_residuals/"
         " (science/<cible>_residuals_<nom>/ pour une correction nommée) :"
         " un fichier par pose, au format de sa source (un t.fits pour"
         " APERO) et sous son nom, dont le flux est remplacé par le résidu"
         " du fit en log, ln f - savgol(ln f) - modèle. Zéro là où le modèle"
         " explique les données, -0,01 là où le flux est 1 % en dessous,"
         " NaN là où le fit n'a donné aucun poids à l'échantillon ; les"
         " autres extensions (longueurs d'onde, blaze) telles quelles. Pas"
         " écrit quand les lignes du fit sont des nuits ou des paquets"
         " (aucune pose n'a son propre résidu). Désactivé par défaut."
         " Demandé à une correction déjà sur disque sans résidus, la"
         " correction est refaite pour les écrire.")),
}


def strings(lang: str) -> dict:
    """The window's words in one language (English for an unknown one)."""
    return dict(FR if str(lang).lower().startswith("fr") else EN)


def option_text(key: str, lang: str, lbl_description: str = "") -> dict:
    """An option's label and help in one language: dict(label, help, lbl).
    `lbl` is LBL's own description (English), given only for an option the
    window has no help of its own for."""
    index = 1 if str(lang).lower().startswith("fr") else 0
    known = OPTIONS.get(key)
    label = known[0][index] if known else key
    helps = known[1] if known else None
    return dict(label=label, help=helps[index] if helps else "",
                lbl="" if helps else (lbl_description or ""))
