--[[

   ultitle

   Underline slide titles.
]]

if quarto.doc.is_format('revealjs') then
   local ultitle

   function Meta(meta)
      ultitle = false
      if meta.ultitle and meta.ultitle == true then
	 ultitle = true
      end
   end

   function Header(el)
      if el.attr.attributes["ultitle"] and el.attr.attributes["ultitle"] == "false" then
	 return el
      end
      if el.level == 2 and ultitle then
       	 table.insert(el.classes, "ultitle")
      end
      return el
   end

   return {
      { Meta = Meta },
      { Header = Header }
   }

end
