using Distributed


# Add this file to following folder:
#-------------------------------------------------------------------------------
# "./.julia/config"
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
addprocs(6)  #ToDo: Modify for the amount of processors you can provide!!!
#-------------------------------------------------------------------------------

#Insert the directory of this project here!
#-------------------------------------------------------------------------------
@everywhere yourdir = "/Desktop/REINFORCEMENT/Master_TRPO_Niko"
#-------------------------------------------------------------------------------



# Include Model Directories
@everywhere begin
	modeldir = homedir()*yourdir
	alldirs = readdir(modeldir)
	for i = 1:length(alldirs)
		alldirs[i] = joinpath(modeldir, alldirs[i], "Julia")

		if isdir(alldirs[i])
			push!(LOAD_PATH, alldirs[i])
		end
	end

	#Include Module Directories
	moduledir = homedir()*yourdir
	alldirs = readdir(moduledir)
	for i = 1:length(alldirs)
		alldirs[i] = joinpath(moduledir, alldirs[i])

		if isdir(alldirs[i])
			push!(LOAD_PATH, alldirs[i])
		end
	end
end

# Set this to have PyPlot as default backend in "Plots.jl"
ENV["PLOTS_DEFAULT_BACKEND"]="PyPlot"
