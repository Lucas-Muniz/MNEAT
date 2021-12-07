using FileIO, JLD2

function addToFile(filename::String, varName::String, var::Any)
	if isfile(filename)
		data = load(filename)
		entry = joinpath("simulation", varName)
		if haskey(data, entry)
			delete!(data, entry)
		end
		data[entry] = var
		save(filename, data)
		#=jldopen(filename, "a+") do file
			entry = joinpath("simulation", varName)
			if haskey(file, entry)
				delete!(file, entry)
			end
			file[entry] = var
			#file["simulation"][varName] = var
		end=#
	else
		jldopen(filename, "w") do file
    		simulation = JLD2.Group(file, "simulation")
    		simulation[varName] = var
		end
	end
end

addToDict(dict::Dict, varName::String, var::Any) = merge(dict, Dict(varName => var))

loadFile(filename::String) = jldopen(filename, "r")["simulation"]

loadFromFile(filename::String, varName::String) = loadFile(filename)[varName]
