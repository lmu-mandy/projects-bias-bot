global ROOT "/Users/vbp/Dropbox/Mac/Documents/Research/NewsMediaBias"

import delimited "$ROOT/data/300-samples.csv", clear
rename answer* *

drop if politics == "Independent" | politics == "Other"
replace newsoutlet = "Fox" if newsoutlet=="FOX"

foreach var in country gender language1 newsoutlet politics biasquestion {
	encode `var', gen("n`var'")
}
replace nbiasquestion = 1 if biasquestion=="is-biased"
replace nbiasquestion = 0 if biasquestion=="is-not-biased"

egen total_bias = total(nbiasquestion), by(npolitics nnewsoutlet)
egen total_obs = count(nbiasquestion), by(npolitics nnewsoutlet)
gen bias_prob = total_bias/total_obs

preserve
gen phase = 2
gen websiteheading = ""
keep phase newsoutlet politics biasquestion websiteheading
save "$ROOT/data/phase3.dta", replace
restore

collapse (mean) meanbias= bias_prob, by(npolitics nnewsoutlet)

generate newsparty = npolitics    if nnewsoutlet == 1
replace  newsparty = npolitics+4  if nnewsoutlet == 2
replace  newsparty = npolitics+8 if nnewsoutlet == 3
sort newsparty

twoway (bar meanbias newsparty if npolitics==2, bcolor(blue)) ///
       (bar meanbias newsparty if npolitics==1, bcolor(red)), ///
       legend(row(1) order(1 "Liberal" 2 "Conservative") ) ///
       xlabel( 1.5 "BBC" 5.5 "CNN" 9.5 "FOX", noticks) ///
       xtitle("") ytitle("Probability of Bias Score")
	   
* Merge with original data set
import delimited "$ROOT/data/phase1.csv", clear
rename answer* *
gen newsoutlet = substr(articlenewsoutlet, 1, 3)
gen phase = 1
keep phase newsoutlet politics biasquestion websiteheading
save "$ROOT/data/phase1.dta", replace

use "$ROOT/data/phase1.dta", clear
append using "$ROOT/data/phase3.dta"
drop if websiteheading!=newsoutlet & phase==1
drop if politics == "Independent" | politics == "Other"

foreach var in newsoutlet politics biasquestion {
	encode `var', gen("n`var'")
}
replace nbiasquestion = 1 if biasquestion=="is-biased"
replace nbiasquestion = 0 if biasquestion=="is-not-biased"

egen total_bias = total(nbiasquestion), by(npolitics nnewsoutlet phase)
egen total_obs = count(nbiasquestion), by(npolitics nnewsoutlet phase)
gen bias_prob = total_bias/total_obs

collapse (mean) bias= bias_prob, by(nnewsoutlet phase npolitics)

// Replace numbers for now
replace bias = .7 if phase==1 & npolitics == 1 & nnewsoutlet == 1
replace bias = .45 if phase==1 & npolitics == 2 & nnewsoutlet == 1
replace bias = .789 if phase==1 & npolitics == 1 & nnewsoutlet == 2
replace bias = .6 if phase==1 & npolitics == 2 & nnewsoutlet == 2
replace bias = .5 if phase==1 & npolitics == 1 & nnewsoutlet == 3
replace bias = .684 if phase==1 & npolitics == 2 & nnewsoutlet == 3

generate newsphase = phase+2*npolitics    if nnewsoutlet == 1
replace  newsphase = phase+2*npolitics+7  if nnewsoutlet == 2
replace  newsphase = phase+2*npolitics+14 if nnewsoutlet == 3
sort newsphase

twoway (bar bias newsphase if phase==2 & npolitics==1, bcolor(red) ylabel(0(.1).8)) ///
	   (bar bias newsphase if phase==2 & npolitics==2, bcolor(blue) ylabel(0(.1).8)) ///
	   (bar bias newsphase if phase==1 & npolitics==1, fcolor(red) fintensity(inten20) bcolor(red) bstyle(histogram) ylabel(0(.1).8)) ///	
	   (bar bias newsphase if phase==1 & npolitics==2, fcolor(blue) fintensity(inten20) bcolor(blue) bstyle(histogram) ylabel(0(.1).8)), ///
       legend(row(1) order(3 "Phase 1" 1 "Phase 2") ) ///
       xlabel( 4.5 `" "Conservative  Liberal     " " " "BBC" "' 11.5 `" "Conservative  Liberal     " " " "CNN" "' 18.5 `" "Conservative  Liberal     " " " "FOX" "', noticks) ///
       xtitle("") ytitle("Probability of Bias Score")

