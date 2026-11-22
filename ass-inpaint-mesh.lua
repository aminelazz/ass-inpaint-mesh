--[[
ass-inpaint-mesh.lua - Aegisub Automation 4 Lua Macro
Erases on-screen text inside a \\clip(m ...) polygon using CPU or GPU inpainting
(OpenCV Telea / Navier-Stokes or LaMa AI) and rebuilds the cleaned background
directly as optimized native ASS \\p1 vector meshes.

Place both `ass-inpaint-mesh.lua` and `rebuilder_worker.py` in your
Aegisub `automation/autoload/` folder (or pick `rebuilder_worker.py` via the
"Browse Python Script..." button; settings persist in `?user/config/ass-inpaint-mesh.json`).
]]

script_name = "Gradient & Background Rebuilder (Inpaint Mesh)"
script_description = "Inpaint text inside \\clip(m ...) and rebuild as optimized ASS \\p1 vector tiles"
script_author = "ass-inpaint-mesh"
script_version = "1.3.0"

local IS_WINDOWS = (package.config:sub(1, 1) == "\\")
local PATH_SEP = IS_WINDOWS and "\\" or "/"
local ok_lfs, lfs = pcall(require, "lfs")

--------------------------------------------------------------------------------
-- Path and File Utilities
--------------------------------------------------------------------------------

local function trim(s)
    if type(s) ~= "string" then
        return ""
    end
    return (s:gsub("^%s+", ""):gsub("%s+$", ""))
end

local function normalize_sep(path)
    if IS_WINDOWS then
        return path:gsub("/", "\\")
    end
    return path:gsub("\\", "/")
end

local function strip_trailing_sep(path)
    if #path > 1 then
        return path:gsub("[/\\]+$", "")
    end
    return path
end

local function dirname(path)
    local clean = strip_trailing_sep(normalize_sep(path))
    local dir = clean:match("^(.*)[/\\][^/\\]+$")
    if not dir or dir == "" then
        return "."
    end
    return dir
end

local function join_path(a, b)
    a = strip_trailing_sep(normalize_sep(a))
    b = normalize_sep(b):gsub("^[/\\]+", "")
    if a == "" or a == "." then
        return b
    end
    return a .. PATH_SEP .. b
end

local function file_exists(path)
    if not path or path == "" then return false end
    local fh = io.open(path, "rb")
    if fh then
        fh:close()
        return true
    end
    return false
end

local function dir_exists(path)
    if not path or path == "" then
        return false
    end
    local clean = strip_trailing_sep(normalize_sep(path))
    if ok_lfs and lfs and lfs.attributes then
        return lfs.attributes(clean, "mode") == "directory"
    end
    local f, err = io.open(clean, "rb")
    if f then
        f:close()
        return false
    end
    if err then
        local lower_err = err:lower()
        return lower_err:match("permission denied") ~= nil or lower_err:match("is a directory") ~= nil
    end
    return false
end

local function ensure_dir(dir_path)
    if dir_exists(dir_path) then
        return true
    end
    local clean = strip_trailing_sep(normalize_sep(dir_path))
    if ok_lfs and lfs and lfs.mkdir then
        local parent = dirname(clean)
        if parent and parent ~= "." and parent ~= clean and not dir_exists(parent) then
            ensure_dir(parent)
        end
        local ok_mk = lfs.mkdir(clean)
        if ok_mk then
            return true
        end
    end
    if IS_WINDOWS then
        local ok = os.execute('mkdir "' .. clean .. '" 2>nul')
        return ok == true or ok == 0
    else
        local ok = os.execute('mkdir -p "' .. clean:gsub('"', '\\"') .. '"')
        return ok == true or ok == 0
    end
end

local function read_all_file(path)
    local fh, err = io.open(path, "rb")
    if not fh then return nil, err end
    local content = fh:read("*a")
    fh:close()
    return content
end

local function write_all_file(path, content)
    local fh, err = io.open(path, "wb")
    if not fh then return false, err end
    fh:write(content)
    fh:close()
    return true
end

local function copy_table(tbl)
    local out = {}
    for k, v in pairs(tbl) do
        out[k] = v
    end
    return out
end

--------------------------------------------------------------------------------
-- Minimal Pure-Lua JSON Encoder & Decoder
--------------------------------------------------------------------------------

local json = {}

local function json_escape_str(s)
    local replacements = {
        ['"']  = '\\"',
        ['\\'] = '\\\\',
        ['\b'] = '\\b',
        ['\f'] = '\\f',
        ['\n'] = '\\n',
        ['\r'] = '\\r',
        ['\t'] = '\\t'
    }
    return '"' .. s:gsub('[%z\1-\31\\"]', function(c)
        return replacements[c] or string.format('\\u%04x', c:byte())
    end) .. '"'
end

local function is_array(tbl)
    local max_idx = 0
    local count = 0
    for k, _ in pairs(tbl) do
        if type(k) ~= "number" or k <= 0 or math.floor(k) ~= k then
            return false
        end
        if k > max_idx then max_idx = k end
        count = count + 1
    end
    return max_idx == count
end

function json.encode(val, pretty, indent_level)
    local t = type(val)
    local lvl = indent_level or 0
    if val == nil then
        return "null"
    elseif t == "boolean" then
        return val and "true" or "false"
    elseif t == "number" then
        if val ~= val or val == math.huge or val == -math.huge then
            return "null"
        end
        if math.floor(val) == val and math.abs(val) < 1e15 then
            return string.format("%.0f", val)
        end
        return tostring(val)
    elseif t == "string" then
        return json_escape_str(val)
    elseif t == "table" then
        if is_array(val) then
            if #val == 0 then
                return "[]"
            end
            local parts = {}
            for i = 1, #val do
                parts[#parts + 1] = json.encode(val[i], pretty, lvl + 1)
            end
            if pretty then
                local pad = string.rep("  ", lvl + 1)
                local close_pad = string.rep("  ", lvl)
                return "[\n" .. pad .. table.concat(parts, ",\n" .. pad) .. "\n" .. close_pad .. "]"
            end
            return "[" .. table.concat(parts, ",") .. "]"
        else
            local keys = {}
            for k, v in pairs(val) do
                if v ~= nil then
                    keys[#keys + 1] = tostring(k)
                end
            end
            if #keys == 0 then
                return "{}"
            end
            table.sort(keys)
            local parts = {}
            for _, k in ipairs(keys) do
                local encoded_v = json.encode(val[k], pretty, lvl + 1)
                local sep = pretty and ": " or ":"
                parts[#parts + 1] = json_escape_str(k) .. sep .. encoded_v
            end
            if pretty then
                local pad = string.rep("  ", lvl + 1)
                local close_pad = string.rep("  ", lvl)
                return "{\n" .. pad .. table.concat(parts, ",\n" .. pad) .. "\n" .. close_pad .. "}"
            end
            return "{" .. table.concat(parts, ",") .. "}"
        end
    else
        error("Unsupported JSON type: " .. t)
    end
end

function json.decode(str)
    local pos = 1
    local len = #str

    local function skip_ws()
        while pos <= len do
            local b = str:byte(pos)
            if b == 32 or b == 9 or b == 10 or b == 13 then
                pos = pos + 1
            else
                break
            end
        end
    end

    local parse_value

    local function parse_string()
        pos = pos + 1 -- skip opening quote
        local buf = {}
        while pos <= len do
            local ch = str:sub(pos, pos)
            if ch == '"' then
                pos = pos + 1
                return table.concat(buf)
            elseif ch == "\\" then
                local esc = str:sub(pos + 1, pos + 1)
                if esc == '"' or esc == "\\" or esc == "/" then
                    buf[#buf + 1] = esc
                    pos = pos + 2
                elseif esc == "b" then buf[#buf + 1] = "\b"; pos = pos + 2
                elseif esc == "f" then buf[#buf + 1] = "\f"; pos = pos + 2
                elseif esc == "n" then buf[#buf + 1] = "\n"; pos = pos + 2
                elseif esc == "r" then buf[#buf + 1] = "\r"; pos = pos + 2
                elseif esc == "t" then buf[#buf + 1] = "\t"; pos = pos + 2
                elseif esc == "u" then
                    local hex = str:sub(pos + 2, pos + 5)
                    local cp = tonumber(hex, 16) or 32
                    if cp < 0x80 then
                        buf[#buf + 1] = string.char(cp)
                    elseif cp < 0x800 then
                        buf[#buf + 1] = string.char(
                            0xC0 + math.floor(cp / 64),
                            0x80 + (cp % 64)
                        )
                    else
                        buf[#buf + 1] = string.char(
                            0xE0 + math.floor(cp / 4096),
                            0x80 + (math.floor(cp / 64) % 64),
                            0x80 + (cp % 64)
                        )
                    end
                    pos = pos + 6
                else
                    buf[#buf + 1] = esc
                    pos = pos + 2
                end
            else
                local next_special = str:find('[\\"]', pos)
                if not next_special then
                    error("Unterminated JSON string")
                end
                buf[#buf + 1] = str:sub(pos, next_special - 1)
                pos = next_special
            end
        end
        error("Unterminated JSON string")
    end

    local function parse_number()
        local num_str = str:match("^-?%d+%.?%d*[eE]?[+-]?%d*", pos)
        if not num_str or num_str == "" then
            error("Invalid JSON number at position " .. pos)
        end
        pos = pos + #num_str
        return tonumber(num_str)
    end

    local function parse_array()
        pos = pos + 1 -- skip '['
        local arr = {}
        skip_ws()
        if str:sub(pos, pos) == "]" then
            pos = pos + 1
            return arr
        end
        while pos <= len do
            arr[#arr + 1] = parse_value()
            skip_ws()
            local sep = str:sub(pos, pos)
            if sep == "]" then
                pos = pos + 1
                return arr
            elseif sep == "," then
                pos = pos + 1
            else
                error("Expected ',' or ']' in JSON array at position " .. pos)
            end
        end
        error("Unterminated JSON array")
    end

    local function parse_object()
        pos = pos + 1 -- skip '{'
        local obj = {}
        skip_ws()
        if str:sub(pos, pos) == "}" then
            pos = pos + 1
            return obj
        end
        while pos <= len do
            skip_ws()
            if str:sub(pos, pos) ~= '"' then
                error("Expected JSON object key string at position " .. pos)
            end
            local k = parse_string()
            skip_ws()
            if str:sub(pos, pos) ~= ":" then
                error("Expected ':' after JSON key at position " .. pos)
            end
            pos = pos + 1
            obj[k] = parse_value()
            skip_ws()
            local sep = str:sub(pos, pos)
            if sep == "}" then
                pos = pos + 1
                return obj
            elseif sep == "," then
                pos = pos + 1
            else
                error("Expected ',' or '}' in JSON object at position " .. pos)
            end
        end
        error("Unterminated JSON object")
    end

    parse_value = function()
        skip_ws()
        local ch = str:sub(pos, pos)
        if ch == '"' then
            return parse_string()
        elseif ch == "{" then
            return parse_object()
        elseif ch == "[" then
            return parse_array()
        elseif ch == "t" and str:sub(pos, pos + 3) == "true" then
            pos = pos + 4
            return true
        elseif ch == "f" and str:sub(pos, pos + 4) == "false" then
            pos = pos + 5
            return false
        elseif ch == "n" and str:sub(pos, pos + 3) == "null" then
            pos = pos + 4
            return nil
        else
            return parse_number()
        end
    end

    return parse_value()
end

--------------------------------------------------------------------------------
-- Worker Detection, Subtitle Helpers & Process Execution
--------------------------------------------------------------------------------

local function get_temp_dir()
    if aegisub and aegisub.decode_path then
        local decoded = aegisub.decode_path("?temp")
        if decoded and decoded ~= "" and decoded ~= "?temp" then
            return decoded:gsub("[/\\]+$", "")
        end
    end
    local env_tmp = os.getenv("TEMP") or os.getenv("TMP") or os.getenv("TMPDIR")
    if env_tmp and env_tmp ~= "" then
        return env_tmp:gsub("[/\\]+$", "")
    end
    return "/tmp"
end

local function make_temp_path(prefix, ext)
    local dir = get_temp_dir()
    local stamp = string.format("%d_%05d", os.time(), math.random(0, 99999))
    return dir .. PATH_SEP .. prefix .. "_" .. stamp .. ext
end

local function locate_worker_script()
    local candidates = {}

    if debug and debug.getinfo then
        local src = debug.getinfo(1, "S").source
        if src and src:sub(1, 1) == "@" then
            local script_path = src:sub(2)
            local dir = script_path:match("^(.*)[/\\][^/\\]+$")
            if dir then
                candidates[#candidates + 1] = join_path(dir, "rebuilder_worker.py")
            end
        end
    end

    if aegisub and aegisub.decode_path then
        candidates[#candidates + 1] = aegisub.decode_path("?user/automation/autoload/rebuilder_worker.py")
        candidates[#candidates + 1] = aegisub.decode_path("?data/automation/autoload/rebuilder_worker.py")
        candidates[#candidates + 1] = aegisub.decode_path("?script/rebuilder_worker.py")
    end

    for _, path in ipairs(candidates) do
        if file_exists(path) then
            return path, candidates
        end
    end

    return nil, candidates
end

local function extract_clip(text)
    if not text or text == "" then return nil end
    -- Search inside ASS override tag blocks {...} to avoid matching plain dialogue text.
    -- Because of the leading backslash, \[Cc][Ll][Ii][Pp] cannot match \iclip.
    for block in text:gmatch("(%b{})") do
        local inner_clip = block:match("\\[Cc][Ll][Ii][Pp]%s*(%b())")
        if inner_clip then
            return "\\clip" .. inner_clip
        end
    end
    return nil
end

local function get_script_resolution(sub)
    local rx, ry, ycbcr = nil, nil, nil
    for i = 1, #sub do
        local line = sub[i]
        if line.class == "info" then
            local key = (line.key or ""):lower()
            if key == "playresx" then
                rx = tonumber(line.value)
            elseif key == "playresy" then
                ry = tonumber(line.value)
            elseif key == "ycbcr matrix" then
                ycbcr = trim(line.value)
            end
        elseif line.class == "dialogue" then
            break
        end
    end
    return rx, ry, ycbcr
end

local function show_error(msg)
    aegisub.dialog.display(
        {{class = "label", label = msg, x = 0, y = 0, width = 1, height = 1}},
        {"OK"},
        {ok = "OK", cancel = "OK"}
    )
end

local function run_command_capture(cmd)
    local full_cmd = cmd .. " 2>&1"
    if IS_WINDOWS then
        -- cmd.exe /c strips the first and last quote only when the whole string is wrapped
        full_cmd = '"' .. full_cmd .. '"'
    end
    local pipe = io.popen(full_cmd, "r")
    if pipe then
        local out = pipe:read("*a") or ""
        pipe:close()
        return out
    end
    local exit_ok = os.execute(full_cmd)
    return tostring(exit_ok)
end

local function invoke_python_worker(python_bin, worker_path, req_path, res_path)
    local configured = trim(python_bin)
    local py_candidates = {}

    if configured ~= "" then
        py_candidates[#py_candidates + 1] = configured
    end
    if configured == "" or configured == "python" or configured == "python3" then
        local fallbacks = IS_WINDOWS
            and {"python", "py -3", "python3"}
            or {"python3", "python"}
        for _, cand in ipairs(fallbacks) do
            if cand ~= configured then
                py_candidates[#py_candidates + 1] = cand
            end
        end
    end

    local last_output = ""
    for _, py_cmd in ipairs(py_candidates) do
        local cmd
        if not py_cmd:find("[/\\]") and py_cmd:find("%s") then
            cmd = string.format('%s "%s" --config "%s" --output "%s"', py_cmd, worker_path, req_path, res_path)
        else
            cmd = string.format('"%s" "%s" --config "%s" --output "%s"', py_cmd, worker_path, req_path, res_path)
        end

        last_output = run_command_capture(cmd)
        if file_exists(res_path) then
            local raw_json = read_all_file(res_path)
            if raw_json and raw_json ~= "" then
                local ok, decoded = pcall(json.decode, raw_json)
                if ok and type(decoded) == "table" then
                    return decoded, last_output
                end
            end
        end
    end

    return nil, last_output
end

--------------------------------------------------------------------------------
-- GUI Configuration & Persistence (?user/config/ass-inpaint-mesh.json)
--------------------------------------------------------------------------------

local ENGINE_ITEMS = {
    "OpenCV Fast (Telea)",
    "OpenCV Smooth (Navier-Stokes)",
    "LaMa AI (CPU - High Quality)",
    "LaMa AI (GPU - High Quality)"
}

local DEVICE_ITEMS = {
    "CPU",
    "GPU (CUDA / MPS)",
    "Auto (Prefer GPU)"
}

local FRAME_MODE_ITEMS = {
    "Single Frame (Current Video Frame)",
    "All Frames in Selected Line"
}

local function contains_item(items, val)
    for _, item in ipairs(items) do
        if item == val then
            return true
        end
    end
    return false
end

local function get_config_file_path()
    if aegisub and aegisub.decode_path then
        local decoded = aegisub.decode_path("?user/config/ass-inpaint-mesh.json")
        if decoded and decoded ~= "" and not decoded:find("^%?user") then
            return normalize_sep(decoded)
        end
    end
    local user_dir
    if IS_WINDOWS then
        local appdata = os.getenv("APPDATA")
        user_dir = (appdata and appdata ~= "") and join_path(appdata, "Aegisub") or "."
    else
        local home = os.getenv("HOME")
        user_dir = (home and home ~= "") and join_path(home, ".aegisub") or "."
    end
    return join_path(join_path(user_dir, "config"), "ass-inpaint-mesh.json")
end

local function load_config(defaults)
    local cfg = copy_table(defaults)
    local path = get_config_file_path()
    local content = read_all_file(path)
    if not content or trim(content) == "" then
        return cfg
    end

    local ok_parse, parsed = pcall(json.decode, content)
    if ok_parse and type(parsed) == "table" then
        for k, default_val in pairs(defaults) do
            if type(parsed[k]) == type(default_val) then
                if k == "script_path" and trim(parsed[k]) == "" then
                    -- Keep auto-detected script path when saved value is blank
                elseif k == "python_bin" and trim(parsed[k]) == "" then
                    -- Keep default python executable when saved value is blank
                elseif k == "engine" and not contains_item(ENGINE_ITEMS, parsed[k]) then
                    -- Ignore unrecognized engine string
                elseif k == "device" and not contains_item(DEVICE_ITEMS, parsed[k]) then
                    -- Ignore unrecognized device string
                elseif k == "frame_mode" and not contains_item(FRAME_MODE_ITEMS, parsed[k]) then
                    -- Ignore unrecognized frame mode string
                elseif k == "overlap_px" then
                    local ov = math.floor(tonumber(parsed[k]) or default_val)
                    cfg[k] = math.max(0, math.min(4, ov))
                else
                    cfg[k] = parsed[k]
                end
            end
        end
    end
    return cfg
end

local function save_config(cfg)
    local path = get_config_file_path()
    ensure_dir(dirname(path))
    return write_all_file(path, json.encode(cfg, true) .. "\n")
end

local function build_defaults()
    local detected_worker = locate_worker_script() or ""
    local defaults = {
        engine = "OpenCV Fast (Telea)",
        device = "CPU",
        grid_size = 4,
        tolerance = 6,
        blur = 0.6,
        enable_overlap = true,
        overlap_px = 1,
        frame_mode = "Single Frame (Current Video Frame)",
        python_bin = "python",
        script_path = detected_worker
    }
    return load_config(defaults)
end

local function make_dialog_config(cfg)
    return {
        {class = "label", label = "Inpainting Engine:", x = 0, y = 0, width = 1, height = 1},
        {
            class = "dropdown",
            name = "engine",
            items = ENGINE_ITEMS,
            value = cfg.engine,
            x = 1, y = 0, width = 3, height = 1
        },

        {class = "label", label = "Compute Device:", x = 0, y = 1, width = 1, height = 1},
        {
            class = "dropdown",
            name = "device",
            items = DEVICE_ITEMS,
            value = cfg.device,
            x = 1, y = 1, width = 3, height = 1
        },

        {class = "label", label = "Grid Resolution (Patch Size in px):", x = 0, y = 2, width = 1, height = 1},
        {class = "intedit", name = "grid_size", value = cfg.grid_size, min = 1, max = 16, x = 1, y = 2, width = 1, height = 1},

        {class = "label", label = "Color Quantization / Merge Tolerance:", x = 0, y = 3, width = 1, height = 1},
        {class = "intedit", name = "tolerance", value = cfg.tolerance, min = 0, max = 30, x = 1, y = 3, width = 1, height = 1},

        {class = "label", label = "Edge Feather / Blur (\\blur):", x = 0, y = 4, width = 1, height = 1},
        {class = "floatedit", name = "blur", value = cfg.blur, min = 0.0, max = 10.0, step = 0.1, x = 1, y = 4, width = 1, height = 1},

        {
            class = "checkbox",
            name = "enable_overlap",
            label = "Overlap Layer Edges + Solid Backing (Anti-Gap for Motion Tracking)",
            value = cfg.enable_overlap,
            x = 0, y = 5, width = 4, height = 1
        },

        {class = "label", label = "Layer Edge Overlap (px):", x = 0, y = 6, width = 1, height = 1},
        {class = "intedit", name = "overlap_px", value = cfg.overlap_px, min = 0, max = 4, x = 1, y = 6, width = 1, height = 1},

        {class = "label", label = "Frame Range Mode:", x = 0, y = 7, width = 1, height = 1},
        {
            class = "dropdown",
            name = "frame_mode",
            items = FRAME_MODE_ITEMS,
            value = cfg.frame_mode,
            x = 1, y = 7, width = 3, height = 1
        },

        {class = "label", label = "Python Executable:", x = 0, y = 8, width = 1, height = 1},
        {class = "edit", name = "python_bin", value = cfg.python_bin, x = 1, y = 8, width = 3, height = 1},

        {class = "label", label = "Python Script Path:", x = 0, y = 9, width = 1, height = 1},
        {class = "edit", name = "script_path", value = cfg.script_path, x = 1, y = 9, width = 3, height = 1}
    }
end

--------------------------------------------------------------------------------
-- Aegisub Automation 4 Macro Entrypoints
--------------------------------------------------------------------------------

function validate_rebuilder(sub, sel, act)
    for _, idx in ipairs(sel) do
        local line = sub[idx]
        if line and line.class == "dialogue" and extract_clip(line.text) ~= nil then
            return true
        end
    end
    return false
end

function run_rebuilder(sub, sel, act)
    math.randomseed(os.time())

    -- Verify video properties
    local proj = aegisub.project_properties()
    local video_file = proj and proj.video_file or ""
    if video_file == "" then
        show_error("No video is loaded in Aegisub.\nLoad the source video first so frames can be inpainted.")
        return
    end
    if video_file:sub(1, 1) == "?" then
        show_error("Dummy videos (?dummy) are not supported.\nPlease load a real video file.")
        return
    end

    local vid_w, vid_h = aegisub.video_size()
    if not vid_w or not vid_h or vid_w <= 0 or vid_h <= 0 then
        show_error("Could not determine loaded video resolution from Aegisub.")
        return
    end

    local script_w, script_h, script_ycbcr = get_script_resolution(sub)
    script_w = script_w or vid_w
    script_h = script_h or vid_h

    -- Collect valid selected lines containing \clip(...)
    local target_indices = {}
    for _, idx in ipairs(sel) do
        local line = sub[idx]
        if line and line.class == "dialogue" and extract_clip(line.text) ~= nil then
            target_indices[#target_indices + 1] = idx
        end
    end
    if #target_indices == 0 then
        show_error("No selected dialogue line contains a \\clip(...) shape.")
        return
    end

    -- Load persisted configuration on script launch
    local cfg = build_defaults()

    -- Display GUI dialog loop with file picker & auto-save on close
    while true do
        local buttons = { "Rebuild", "Browse Python Script...", "Cancel" }
        local button_ids = { ok = "Rebuild", cancel = "Cancel" }
        local btn, res = aegisub.dialog.display(make_dialog_config(cfg), buttons, button_ids)

        if type(res) == "table" then
            for k, v in pairs(res) do
                cfg[k] = v
            end
        end

        if not btn or btn == "Cancel" then
            save_config(cfg)
            return
        elseif btn == "Browse Python Script..." then
            local default_py_dir = ""
            local cur_py = trim(cfg.script_path)
            if cur_py ~= "" then
                if file_exists(cur_py) then
                    default_py_dir = dirname(cur_py)
                elseif dir_exists(cur_py) then
                    default_py_dir = cur_py
                end
            end

            local chosen_py = aegisub.dialog.open(
                "Select Python script (rebuilder_worker.py)",
                default_py_dir,
                "",
                "Python scripts (*.py)|*.py|All files (*.*)|*.*",
                false,
                true
            )
            if chosen_py and chosen_py ~= "" then
                cfg.script_path = chosen_py
            end
            save_config(cfg)
        elseif btn == "Rebuild" then
            cfg.script_path = trim(cfg.script_path)
            cfg.python_bin = trim(cfg.python_bin)
            if cfg.python_bin == "" then
                cfg.python_bin = "python"
            end

            if cfg.script_path == "" or not file_exists(cfg.script_path) then
                local _, searched_paths = locate_worker_script()
                save_config(cfg)
                show_error(
                    "Could not find Python worker script:\n"
                    .. (cfg.script_path ~= "" and cfg.script_path or "(empty)")
                    .. "\n\nUse 'Browse Python Script...' to select rebuilder_worker.py.\nSearched defaults:\n- "
                    .. table.concat(searched_paths, "\n- ")
                )
            else
                save_config(cfg)
                break
            end
        end
    end

    local worker_path = cfg.script_path

    -- Sort indices ascending and iterate in reverse so insertions never shift earlier indices
    table.sort(target_indices)
    local all_warnings = {}
    local stale_worker_warned = false
    local stale_worker_msg = nil

    for pos_i = #target_indices, 1, -1 do
        if aegisub.progress.is_cancelled() then
            break
        end

        local line_idx = target_indices[pos_i]
        local guide_line = sub[line_idx]
        local clip_tag = extract_clip(guide_line.text)

        aegisub.progress.title(script_name)
        aegisub.progress.task(string.format("Rebuilding line %d (%d/%d)...", line_idx, #target_indices - pos_i + 1, #target_indices))
        aegisub.progress.set(((#target_indices - pos_i) / #target_indices) * 100)

        local start_frame = aegisub.frame_from_ms(guide_line.start_time)
        local end_frame_raw = aegisub.frame_from_ms(guide_line.end_time)
        if start_frame == nil then
            save_config(cfg)
            show_error("Could not map line timing to video frames. Ensure video timecodes are loaded.")
            return
        end

        local frames_payload = {}
        if cfg.frame_mode == "All Frames in Selected Line" then
            local end_frame = (end_frame_raw or (start_frame + 1)) - 1
            if end_frame < start_frame then
                end_frame = start_frame
            end
            for f = start_frame, end_frame do
                local f_start = aegisub.ms_from_frame(f) or guide_line.start_time
                local f_end = aegisub.ms_from_frame(f + 1) or guide_line.end_time
                if f == start_frame then f_start = guide_line.start_time end
                if f == end_frame then f_end = guide_line.end_time end
                frames_payload[#frames_payload + 1] = {
                    frame = f,
                    start_ms = f_start,
                    end_ms = f_end
                }
            end
        else
            local cur_frame = proj.video_position
            if cur_frame == nil or cur_frame < 0 then
                cur_frame = start_frame
            end
            frames_payload[1] = {
                frame = cur_frame,
                start_ms = guide_line.start_time,
                end_ms = guide_line.end_time
            }
        end

        local req_path = make_temp_path("ass_inpaint_req", ".json")
        local res_path = make_temp_path("ass_inpaint_res", ".json")

        local job_payload = {
            video_path = video_file,
            video_width = vid_w,
            video_height = vid_h,
            script_width = script_w,
            script_height = script_h,
            clip = clip_tag,
            engine = cfg.engine,
            device = cfg.device,
            grid_size = cfg.grid_size,
            tolerance = cfg.tolerance,
            blur = cfg.blur,
            enable_overlap = cfg.enable_overlap,
            overlap_px = (cfg.enable_overlap and cfg.overlap_px or 0),
            dilate = 3,
            ycbcr_matrix = script_ycbcr or "",
            base_layer = guide_line.layer or 0,
            style = guide_line.style or "Default",
            actor = guide_line.actor or "",
            frames = frames_payload
        }

        local wrote_ok, write_err = write_all_file(req_path, json.encode(job_payload))
        if not wrote_ok then
            save_config(cfg)
            show_error("Failed to write temporary request JSON:\n" .. tostring(write_err))
            return
        end

        local worker_res, raw_log = invoke_python_worker(cfg.python_bin, worker_path, req_path, res_path)
        os.remove(req_path)
        os.remove(res_path)

        if not worker_res then
            save_config(cfg)
            show_error(
                "Python worker failed to produce a valid result.\n\n"
                .. "Make sure Python and requirements.txt packages (numpy, opencv-python, Pillow) are installed.\n\n"
                .. "Worker output:\n" .. tostring(raw_log)
            )
            return
        end

        if worker_res.status ~= "ok" then
            save_config(cfg)
            show_error("Worker returned an error:\n" .. tostring(worker_res.error or raw_log))
            return
        end

        if not stale_worker_warned and cfg.enable_overlap and (tonumber(cfg.overlap_px) or 0) > 0 then
            local st = type(worker_res.stats) == "table" and worker_res.stats or nil
            if worker_res.worker_version == nil or st == nil or st.backing_layer ~= true or (tonumber(st.overlap_px) or 0) <= 0 then
                stale_worker_warned = true
                stale_worker_msg = "Detected an older rebuilder_worker.py without v1.3.0 anti-gap solid backing & overlap support:\n"
                    .. worker_path
                    .. "\n\nPlease replace it with the updated rebuilder_worker.py (v1.3.0+) or pick the updated script via 'Browse Python Script...'."
                all_warnings[#all_warnings + 1] = stale_worker_msg
            end
        end

        if worker_res.warnings then
            for _, w in ipairs(worker_res.warnings) do
                all_warnings[#all_warnings + 1] = w
            end
        end

        local events = worker_res.events or {}
        if #events > 0 then
            -- Comment out the original guide line and insert generated patch lines immediately above it
            local commented_guide = copy_table(guide_line)
            commented_guide.comment = true
            sub[line_idx] = commented_guide

            for k, ev in ipairs(events) do
                local patch_line = copy_table(guide_line)
                patch_line.comment = false
                patch_line.layer = tonumber(ev.layer) or ((guide_line.layer or 0) + k - 1)
                if ev.start_ms ~= nil then patch_line.start_time = tonumber(ev.start_ms) end
                if ev.end_ms ~= nil then patch_line.end_time = tonumber(ev.end_ms) end
                patch_line.text = ev.text
                sub.insert(line_idx + k - 1, patch_line)
            end
        end
    end

    save_config(cfg)
    aegisub.progress.set(100)
    aegisub.set_undo_point(script_name)

    if stale_worker_msg ~= nil then
        show_error(stale_worker_msg)
    end

    if #all_warnings > 0 then
        aegisub.log("[ASS Inpaint Mesh] Warnings:\n- " .. table.concat(all_warnings, "\n- ") .. "\n")
    end
end

aegisub.register_macro(script_name, script_description, run_rebuilder, validate_rebuilder)
