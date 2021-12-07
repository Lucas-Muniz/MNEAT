# track count of various global variables & holds refernce to the config
mutable struct Global
    speciesCnt::Int64
    chromosomeCnt::Int64
    nodeCnt::Int64
    layerCnt::Int64
    innov_number::Int64
    #innovations::Dict{(Int64,Int64),Int64}
    cf::Config
    function Global(cf::Config)
        new(0,0,0,0,0,cf) # global dictionary
    end
end
