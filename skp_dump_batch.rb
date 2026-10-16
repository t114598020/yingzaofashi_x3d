# Run inside SketchUp 8 (-RubyStartup). For every line "in.skp<TAB>out.skpdump" of ENV['SKP_LIST']:
# opens the model and writes the same dump as skp_dump.rb. Progress goes to SKP_LIST + '.log';
# SKP_LIST + '.done' is written at the end.
LIST = ENV['SKP_LIST']
JOBS = File.readlines(LIST).map { |l| l.chomp.split("\t") }.select { |a| a.length == 2 }

def log(msg)
  File.open(LIST + '.log', 'a') { |f| f.puts "#{Time.now.strftime('%H:%M:%S')} #{msg}" }
end

def dump_model(out)
  m = Sketchup.active_model
  ids = {}
  m.definitions.each_with_index { |d, i| ids[d] = i + 1 }
  File.open(out + '.part', 'w') do |f|
    write = lambda do |id, name, ents|
      f.puts "D\t#{id}\t#{name}"
      ents.each do |e|
        if e.is_a?(Sketchup::Face)
          next unless e.visible? && e.layer.visible?
          n = e.normal
          mesh = e.mesh(0)
          mesh.polygons.each do |poly|
            pts = poly.map { |k| mesh.point_at(k.abs) }
            f.puts 'F ' + (pts.map { |p| [p.x.to_f, p.y.to_f, p.z.to_f] }.flatten + [n.x, n.y, n.z]).map { |x| '%.6f' % x }.join(' ')
          end
        elsif e.is_a?(Sketchup::ComponentInstance) || e.is_a?(Sketchup::Group)
          next unless e.visible? && e.layer.visible?
          d = e.is_a?(Sketchup::Group) ? e.entities.parent : e.definition
          f.puts "C\t#{ids[d]}\t" + e.transformation.to_a.map { |x| '%.9g' % x }.join(' ')
        end
      end
    end
    write.call(0, 'model', m.entities)
    m.definitions.each { |d| write.call(ids[d], d.name, d.entities) unless d.image? }
  end
  File.rename(out + '.part', out)
end

def run(k)
  if k >= JOBS.length
    File.open(LIST + '.done', 'w') { |f| f.puts 'ok' }
    return
  end
  src, out = JOBS[k]
  if File.exist?(out)
    run(k + 1)
    return
  end
  log "open #{src}"
  begin
    Sketchup.open_file(src)
  rescue Exception => ex
    log "open failed #{src}: #{ex.message}"
  end
  UI.start_timer(1, false) do
    begin
      dump_model(out)
      log "ok #{out}"
    rescue Exception => ex
      log "failed #{out}: #{ex.message}"
    end
    run(k + 1)
  end
end

UI.start_timer(2, false) { run(0) }
