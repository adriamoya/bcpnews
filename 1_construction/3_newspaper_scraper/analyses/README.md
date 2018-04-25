# File descriptions

* __train.json__: the training set
* __test.json__: the test set

# Data fields

* __authors__: author(s) of the article (array)
* __date__: date of article publication (datetime)
* __day_of_week__: day of the week of article publication (string)
* __domain__: newspaper website domain
* __flag__: target variable (`0`: not included in BBB newsletter, `1`: included)
* __keywords__: list of keywords suggested by nlp application (array)
* __section__: newspaper section
* __summary__: brief summary of the article
* __text__: body text of the article
* __title__: title of the article
* __url__: url link to the article

# Example

```json
{
  "authors":["Eduardo Segovia","José Antonio Navas","E. Sanz","M. Valero","Contacta Al Autor"],
  "date":1478304000000,
  "day_of_week":"Saturday",
  "domain":"elconfidencial.com",
  "flag":1,
  "keywords":["todos","inmuebles","para","retasar","pisos","sus","en","y","banco","del","el","que","los","vivienda","se","las","rebajas","la","noticias"],
  "section":"vivienda",
  "summary":"Malos tiempos para encontrar un chollo entre los pisos que tienen a la venta los bancos.\nEsta norma obliga a volver a tasar (y provisionar) todos los inmuebles que tenga el banco en balance con el descuento aplicado a las ventas, por lo que cualquier alegría en los precios puede tener un enorme impacto en sus cuentas.\nLas provisiones es dinero que los bancos apartan para cubrir el posible impago de los créditos o pérdida de valor de inmuebles, acciones o bonos.\nNo se trata de una norma estándar para todo el sector, ya que en esto se aplican los llamados \"modelos internos\", que son diferentes para cada entidad.\nEn todo caso, con esta norma se refuerza la idea de que cuanto mejor estén provisionados los inmuebles, más fácil será venderlos.",
  "text":"Malos tiempos para encontrar un chollo entre los pisos que tienen a la venta los bancos. El interés de algunos por librarse de las viviendas adjudicadas les ha llevado a ofrecer rebajas muy interesantes en el pasado, pero esta práctica se ha acabado con la nueva circular contable del Banco de España que acaba de entrar en vigor. Esta norma obliga a volver a tasar (y provisionar) todos los inmuebles que tenga el banco en balance con el descuento aplicado a las ventas, por lo que cualquier alegría en los precios puede tener un enorme impacto en sus cuentas.\n\nHasta ahora, cuando un banco vendía un piso por debajo de su valor de tasación, tenía la obligación de provisionar la diferencia con el precio al que lo tuviera valorado en su balance (valor en libros) y apuntarse la pérdida correspondiente. Pero solo para ese inmueble individual. Las provisiones es dinero que los bancos apartan para cubrir el posible impago de los créditos o pérdida de valor de inmuebles, acciones o bonos. Este dinero resta del beneficio de la entidad, o lo que es lo mismo, supone una pérdida.\n\nEl gobernador del Banco de España, Luis Linde (EFE)\n\nLo que cambia la nueva circular es que, a partir de ahora, esa pérdida no solo se referirá a cada piso individual, sino que esa rebaja tendrá que aplicarse a todos los inmuebles similares que tenga el banco en su balance. Y, en consecuencia, deberá provisionarse la pérdida de valor de todos ellos, y dado que la banca todavía tiene activos adjudicados por valor de 81.500 millones de euros (solo se han reducido un 4% desde 2011), estamos hablando de un impacto potencial muy importante en sus resultados.\n\nNo se trata de una norma estándar para todo el sector, ya que en esto se aplican los llamados \"modelos internos\", que son diferentes para cada entidad. Y algunos toman como referencia las ventas realizadas en los últimos tres meses, otros las de los últimos seis... Pero lo que es obligatorio es que estos modelos incluyan la referencia de las operaciones realizadas en los meses anteriores, según fuentes del sector. La justificación de esta medida es que los inmuebles deben estar valorado a precios de mercado y que ese precio debe ser similar para todos los de la misma entidad; por tanto, si vende algunos con un descuento, ese descuento debe aplicarse a todos.\n\nEl BdE critica que los bancos no vendan pisos\n\nResulta un tanto contradictorio que el mismo Banco de España que ha establecido esta norma -que entró en vigor en octubre y se aplicará ya a los resultados de cierre del año- critique en su último Informe de Estabilidad Financiera que los bancos no den salida a sus adjudicados con mayor celeridad: \"En el último año, este importe de activos improductivos [incluye los préstamos morosos] se ha reducido en un 12 %, si bien aún representa un porcentaje significativo del activo total de los bancos en su negocio en España y constituye un elemento de presión negativo sobre la cuenta de resultados y la rentabilidad de las entidades\".\n\nSede del Banco de España, en la Plaza de Cibeles en Madrid (EFE)\n\nComo adelantó El Confidencial, la nueva norma también obliga a las entidades a volver a tasar sus inmuebles adjudicados todos los años, en vez de cada tres ejercicios como sucedía hasta ahora. Esto también pretende que los bancos tengan su ladrillo valorado a niveles realistas y, de nuevo, que doten las provisiones necesarias para cubrir la diferencia con el valor al que se lo adjudicaron. En la presentación de los resultados del tercer trimestre, la mayoría de los grandes bancos explicaron que la nueva circular significará un trasvase de provisiones desde el crédito moroso a los adjudicados, pero que el efecto neto será mínimo.\n\nEn todo caso, con esta norma se refuerza la idea de que cuanto mejor estén provisionados los inmuebles, más fácil será venderlos. Esa es una de las razones que explican las dudas del mercado sobre la capacidad del Banco Popular de deshacerse de 15.000 millones de adjudicados en dos años, ya que su tasa de cobertura con provisiones se sitúa en el 36%, frente a una media del 50% en el sector.",
  "title":"Rebajas: Los bancos acaban con los chollos de pisos para no retasar todos sus inmuebles. Noticias de Vivienda",
  "url":"http:\/\/www.elconfidencial.com\/vivienda\/2016-11-05\/bancos-rebajar-pisos-retasar-inmuebles-adjudicados-circular-contable_1285149\/"
}

```
