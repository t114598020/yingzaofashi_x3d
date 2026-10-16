# Run inside SketchUp 8 (-RubyStartup). Opens SKP_IN and writes every component definition once:
#   D <id> <name>               definition (id 0 = the model itself)
#   F x y z x y z x y z nx ny nz  one triangle of a face in the definition (inches), face normal
#   C <child id> <16 numbers>   a placed child component/group, SketchUp Transformation#to_a (column-major)
# then writes DUMP_OUT + '.done'.
SKP_IN = ENV['SKP_IN']
DUMP_OUT = ENV['SKP_OUT']

def dump_model
  m = Sketchup.active_model
  ids = {}
  m.definitions.each_with_index { |d, i| ids[d] = i + 1 }
  File.open(DUMP_OUT, 'w') do |f|
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
  File.open(DUMP_OUT + '.done', 'w') { |f| f.puts 'ok' }
end

UI.start_timer(2, false) do
  Sketchup.open_file(SKP_IN)
  UI.start_timer(3, false) do
    begin
      dump_model
    rescue Exception => ex
      File.open(DUMP_OUT + '.done', 'w') { |f| f.puts ex.message; f.puts ex.backtrace.join("\n") }
    end
  end
end
